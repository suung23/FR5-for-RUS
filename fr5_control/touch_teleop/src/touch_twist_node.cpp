#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32.hpp> // 그리퍼 제어용 메시지
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <HD/hd.h>
#include <HDU/hduVector.h>
#include <HDU/hduError.h>

#include <mutex>
#include <memory>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

// 그리퍼 지령 값. 기동 로그가 "GRIP (30)" 이라 말하면서 실제로는 5.0 을 발행하고
// 있었다. 상수 하나로 묶어 로그와 동작이 갈라지지 않게 한다.
static constexpr double GRIPPER_CLOSED = 5.0;
static constexpr double GRIPPER_OPEN = 100.0;

class TouchTeleopNode : public rclcpp::Node {
public:
    struct TouchState {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position; 
        Eigen::Matrix3d rotation; 
        HDint buttons = 0; 
        bool valid = false;
        
        // Velocity filtering state
        Eigen::Vector3d filtered_lin_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d filtered_ang_vel = Eigen::Vector3d::Zero();
        rclcpp::Time last_time;
        
        // Previous state for velocity calculation
        Eigen::Vector3d prev_position;
        Eigen::Matrix3d prev_rotation;
        
        bool first_run = true;

        // Button state tracking
        bool was_btn2_pressed = false;
        bool was_deadman_engaged = false;
    };

    struct DeviceContext {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::string name;
        HHD hHD;
        
        // Each device has its own publishers
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gripper_pub;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr stylus_pub;

        TouchState state;
        std::string prefix; // "touch/left" or "touch/right"
    };

    // 프로파일 하나가 스케일·필터·데드존을 통째로 묶는다. 복강경 자유공간 조작과
    // 초음파 접근 조작은 요구 분해능이 자릿수로 다르므로 값 하나만 바꿔서는 안 된다.
    struct Profile {
        double linear_scale;
        double angular_scale;
        double filter_alpha;
        double linear_deadzone;   // m/s, 필터 후 body 속도에 적용
        double angular_deadzone;  // rad/s, 위와 같음
        double lin_sign[3];       // 축별 부호 뒤집개 (프로파일 무관)
        double ang_sign[3];
        double lin_gain[3];       // 축별 병진 이득, **장치 world 기준** (프로파일 무관)
        double tip_roll_deg;      // 침투축 둘레 회전 (프레임 정합의 남은 자유도)
    };

    TouchTeleopNode() : Node("touch_teleop_node") {
        // --- 프로파일 선택 (런타임 변경 가능) ---
        // 매 주기 읽으므로 `ros2 param set` 으로 조작 중에도 바꿀 수 있다.
        this->declare_parameter<std::string>("teleop.profile", "laparoscopic");

        // 복강경 자유공간 조작. 기존 값 그대로.
        this->declare_parameter<double>("teleop.laparoscopic.linear_scale", 0.6);
        this->declare_parameter<double>("teleop.laparoscopic.angular_scale", 0.6);
        this->declare_parameter<double>("teleop.laparoscopic.filter_alpha", 0.4);
        this->declare_parameter<double>("teleop.laparoscopic.linear_deadzone", 0.001);
        this->declare_parameter<double>("teleop.laparoscopic.angular_deadzone", 0.1);

        // 초음파 접근 조작 (DESIGN_NOTES §12.2 TELEOP_APPROACH).
        // 회전 데드존이 핵심이다. 0.1 rad/s 로는 RUS 각속도 상한 0.2 rad/s 에 대해
        // 사용 가능 구간이 3:1 밖에 안 나와 프로브를 미세하게 기울일 수 없다.
        this->declare_parameter<double>("teleop.us_approach.linear_scale", 0.08);
        this->declare_parameter<double>("teleop.us_approach.angular_scale", 0.15);
        this->declare_parameter<double>("teleop.us_approach.filter_alpha", 0.25);
        this->declare_parameter<double>("teleop.us_approach.linear_deadzone", 0.002);
        this->declare_parameter<double>("teleop.us_approach.angular_deadzone", 0.02);

        // 자유공간 검증용. 프로브도 조직도 없는 단계에서 접촉용 스케일(us_approach)을
        // 쓸 이유가 없다. 데드존은 손 잡음 특성이라 us_approach 와 같은 값을 쓴다.
        this->declare_parameter<double>("teleop.freespace.linear_scale", 0.4);
        this->declare_parameter<double>("teleop.freespace.angular_scale", 0.4);
        this->declare_parameter<double>("teleop.freespace.filter_alpha", 0.3);
        this->declare_parameter<double>("teleop.freespace.linear_deadzone", 0.002);
        this->declare_parameter<double>("teleop.freespace.angular_deadzone", 0.05);

        // RUS 는 오른팔 단일 제어다 (DESIGN_NOTES: 제어 대상 오른팔 192.168.58.3).
        // 왼쪽은 비활성, 오른쪽은 기본 장치. 위 initDevice 의 규약 참조.
        // 축별 부호 뒤집개. 매 주기 읽으므로 조작하면서 확정할 수 있다:
        //   ros2 param set /touch_teleop_node teleop.angular_sign "[1.0, -1.0, 1.0]"
        // 아래 매핑에 곱해지기만 한다. 기본 [1,1,1] 이면 매핑 그대로다.
        // 침투축(프로브 +z) 둘레 회전. 스타일러스 촉 ≡ 프로브 침투축 을 고정해도
        // 그 축 둘레 회전은 남는다 — 영상면이 플랜지에 대해 어떻게 놓이는지는
        // 프로브가 장착되어야 정해지기 때문이다. 90 의 배수를 넣으면 축 정렬이
        // 유지되고, 90° 는 pitch 와 roll 이 서로 바뀐 것으로 느껴진다.
        // 매 주기 읽으므로 조작 중에 바꿔가며 확정할 수 있다.
        this->declare_parameter<double>("teleop.tip_roll_deg", 90.0);

        this->declare_parameter<std::vector<double>>("teleop.linear_sign", {1.0, 1.0, 1.0});
        this->declare_parameter<std::vector<double>>("teleop.angular_sign", {1.0, 1.0, 1.0});

        // 축별 병진 이득 [좌우, 상하, 앞뒤]. linear_scale 위에 곱해진다.
        // linear_sign 과 달리 프로브 축이 아니라 **장치 world 축**(조작자의 손 기준)에
        // 걸린다 — 자세한 근거는 아래 processDevice 의 주석. 매 주기 읽으므로
        // 조작하면서 확정할 수 있다:
        //   ros2 param set /touch_teleop_node teleop.linear_axis_gain "[1.0, 2.0, 2.0]"
        this->declare_parameter<std::vector<double>>("teleop.linear_axis_gain", {1.0, 1.0, 1.0});

        this->declare_parameter<std::string>("left_dev_name", "");
        this->declare_parameter<std::string>("right_dev_name", "default");

        std::string left_name = this->get_parameter("left_dev_name").as_string();
        std::string right_name = this->get_parameter("right_dev_name").as_string();

        // Initialize Devices
        // Left
        initDevice(left_name, "left");
        // Right
        initDevice(right_name, "right");

        if (devices_.empty()) {
             RCLCPP_ERROR(this->get_logger(), "No haptic devices initialized. Please check device names.");
             throw std::runtime_error("No haptic devices found");
        }

            // --- Start Scheduler ---
            hd_scheduler_handle_ = hdScheduleAsynchronous(touchCallback, this, HD_MAX_SCHEDULER_PRIORITY);
            hdStartScheduler();
            
            timer_ = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&TouchTeleopNode::timerCallback, this));
            
            const Profile p = loadProfile();
            RCLCPP_INFO(this->get_logger(), "Touch Teleop Ready.");
            RCLCPP_INFO(this->get_logger(), " - Button 1 (Gray): 누르고 있는 동안에만 동작 (데드맨). 놓으면 정지");
            RCLCPP_INFO(this->get_logger(), " - Button 2 (White): Hold to GRIP (%.0f), Release to OPEN (%.0f)",
                        GRIPPER_CLOSED, GRIPPER_OPEN);
            RCLCPP_INFO(this->get_logger(), "Active Devices: %zu", devices_.size());
            RCLCPP_INFO(this->get_logger(),
                "Profile '%s': lin x%.3f (deadzone %.4f m/s), ang x%.3f (deadzone %.4f rad/s), alpha %.2f",
                this->get_parameter("teleop.profile").as_string().c_str(),
                p.linear_scale, p.linear_deadzone, p.angular_scale, p.angular_deadzone, p.filter_alpha);
            RCLCPP_INFO(this->get_logger(),
                "프레임 정합: 촉(-Z) ≡ 침투(+z), 침투축 둘레 %.0f°", p.tip_roll_deg);
            RCLCPP_INFO(this->get_logger(),
                "축별 병진 이득 (장치 world, 좌우/상하/앞뒤): x%.2f / x%.2f / x%.2f",
                p.lin_gain[0], p.lin_gain[1], p.lin_gain[2]);
    }

    ~TouchTeleopNode() {
        if (!devices_.empty()) {
            hdStopScheduler();
            hdUnschedule(hd_scheduler_handle_);
        }
        // hdDisableDevice is not strictly needed if we just exit, 
        // but cleaner to maybe disable each? 
        // hdDisableDevice can only be called if that context is current or we iterate.
        // Usually fine to just stop scheduler.
    }

private: 
    std::vector<std::shared_ptr<DeviceContext>> devices_;
    std::mutex state_mutex_;

    HDSchedulerHandle hd_scheduler_handle_;
    rclcpp::TimerBase::SharedPtr timer_;

    // dev_name 규약:
    //   ""        → 이 쪽 장치를 아예 열지 않는다 (RUS 는 오른팔 단일)
    //   "default" → HD_DEFAULT_DEVICE(NULL) 로 기본 장치를 연다
    //   그 외      → Touch_Setup 으로 등록된 이름으로 연다
    //
    // USB Touch X 는 HID 로 잡히고, HID 장치는 Touch_Setup 을 거치지 않으므로
    // 등록된 이름이 존재하지 않는다. 이름을 넘기면 무조건 실패한다. "default" 가
    // 이 경우의 정답이다. 이더넷 장치는 이름 등록이 있으므로 기존 경로도 남긴다.
    void initDevice(const std::string& dev_name, const std::string& side) {
        if (dev_name.empty()) {
            RCLCPP_INFO(this->get_logger(), "%s 장치는 비활성 (이름이 비어 있음)", side.c_str());
            return;
        }

        const bool use_default = (dev_name == "default");
        const char *hd_name = use_default ? HD_DEFAULT_DEVICE : dev_name.c_str();
        const char *label = use_default ? "<default>" : dev_name.c_str();

        HHD hHD = hdInitDevice(hd_name);

        // hdGetError() 를 먼저 부른다. 예전 코드는
        //   hHD == HD_INVALID_HANDLE || HD_DEVICE_ERROR(error = hdGetError())
        // 였는데, 핸들이 무효면 단축평가로 error 가 대입되지 않은 채 출력되어
        // 실제 원인 대신 "No error" 가 찍혔다.
        HDErrorInfo error = hdGetError();
        if (hHD == HD_INVALID_HANDLE || HD_DEVICE_ERROR(error)) {
            RCLCPP_WARN(this->get_logger(), "Failed to init device '%s' (%s): %s",
                        label, side.c_str(), hdGetErrorString(error.errorCode));
            return;
        }

        auto ctx = std::make_shared<DeviceContext>();
        ctx->name = dev_name;
        ctx->hHD = hHD;
        ctx->prefix = "touch/" + side; // e.g., touch/left

        // Create publishers with specific topics
        std::string twist_topic = "/fr5_" + side + "/desired_twist";
        std::string gripper_topic = "/fr5_" + side + "/desired_gripper_pose";

        ctx->twist_pub = this->create_publisher<geometry_msgs::msg::Twist>(twist_topic, 10);
        ctx->gripper_pub = this->create_publisher<std_msgs::msg::Float32>(gripper_topic, 10);
        ctx->stylus_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/touch/" + side + "/stylus_pose", 10);

        // Init state time
        ctx->state.last_time = this->now();

        devices_.push_back(ctx);
        RCLCPP_INFO(this->get_logger(), "Initialized %s on topics: %s, %s", label, twist_topic.c_str(), gripper_topic.c_str());
    }

    // --- Haptic Loop (1kHz) ---
    static HDCallbackCode HDCALLBACK touchCallback(void *pUserData) {
        TouchTeleopNode *node = static_cast<TouchTeleopNode *>(pUserData);
        
        for (auto &ctx : node->devices_) {
            hdMakeCurrentDevice(ctx->hHD);
            hdBeginFrame(ctx->hHD);
            
            double raw_transform[16];
            hdGetDoublev(HD_CURRENT_TRANSFORM, raw_transform);
            
            HDint buttons = 0;
            hdGetIntegerv(HD_CURRENT_BUTTONS, &buttons); // bitmask

            hdEndFrame(ctx->hHD);

            Eigen::Map<Eigen::Matrix4d> mat(raw_transform);

            {
                std::lock_guard<std::mutex> lock(node->state_mutex_);
                ctx->state.position = mat.block<3, 1>(0, 3) * 0.001; 
                ctx->state.rotation = mat.block<3, 3>(0, 0);
                ctx->state.buttons  = buttons;
                ctx->state.valid = true;
            }
        }

        return HD_CALLBACK_CONTINUE;
    }

    // 알 수 없는 프로파일 이름은 조용히 넘기지 않는다. 접근 프로파일을 의도했는데
    // 오타로 복강경 스케일(7배 빠름)이 도는 것이 가장 위험하다.
    Profile loadProfile() {
        const std::string name = this->get_parameter("teleop.profile").as_string();
        std::string key;
        if (name == "laparoscopic" || name == "us_approach" || name == "freespace") {
            key = name;
        } else {
            RCLCPP_ERROR(this->get_logger(),
                "알 수 없는 teleop.profile '%s'. 가장 보수적인 us_approach 로 대체한다. "
                "'laparoscopic' 또는 'us_approach' 여야 한다.", name.c_str());
            key = "us_approach";
        }
        const std::string p = "teleop." + key + ".";
        Profile out{
            this->get_parameter(p + "linear_scale").as_double(),
            this->get_parameter(p + "angular_scale").as_double(),
            this->get_parameter(p + "filter_alpha").as_double(),
            this->get_parameter(p + "linear_deadzone").as_double(),
            this->get_parameter(p + "angular_deadzone").as_double(),
            {1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0},
            this->get_parameter("teleop.tip_roll_deg").as_double(),
        };
        const auto ls = this->get_parameter("teleop.linear_sign").as_double_array();
        const auto as = this->get_parameter("teleop.angular_sign").as_double_array();
        const auto lg = this->get_parameter("teleop.linear_axis_gain").as_double_array();
        for (size_t i = 0; i < 3; ++i) {
            if (i < ls.size()) out.lin_sign[i] = ls[i];
            if (i < as.size()) out.ang_sign[i] = as[i];
            if (i < lg.size()) out.lin_gain[i] = lg[i];
        }
        return out;
    }

    // --- ROS Loop (50Hz) ---
    void timerCallback() {
        rclcpp::Time now = this->now();

        // 매 주기 읽는다 — 조작 중 `ros2 param set` 으로 즉시 반영되게.
        const Profile profile = loadProfile();

        std::lock_guard<std::mutex> lock(state_mutex_);

        for (auto &ctx : devices_) {
            if (!ctx->state.valid) continue;

            processDevice(ctx, now, profile);
        }
    }

    void processDevice(std::shared_ptr<DeviceContext> ctx, rclcpp::Time now,
                       const Profile &profile)
    {
        const double l_scale = profile.linear_scale;
        const double a_scale = profile.angular_scale;
        const double alpha   = profile.filter_alpha;
        TouchState &s = ctx->state;

        if (s.first_run) {
            s.prev_position = s.position;
            s.prev_rotation = s.rotation;
            s.last_time = now;
            s.first_run = false;
            return;
        }

        // =========================================================
        // 1. GRIPPER CONTROL (Button 2)
        // =========================================================
        bool is_btn2_pressed = (s.buttons & HD_DEVICE_BUTTON_2) != 0;
        
        std_msgs::msg::Float32 gripper_msg;
        gripper_msg.data = is_btn2_pressed ? GRIPPER_CLOSED : GRIPPER_OPEN;
        ctx->gripper_pub->publish(gripper_msg);

        if (is_btn2_pressed != s.was_btn2_pressed) {
            // Log with device prefix
            RCLCPP_INFO(this->get_logger(), "[%s] Button 2 Changed! Publishing gripper: %f", ctx->prefix.c_str(), gripper_msg.data);
            s.was_btn2_pressed = is_btn2_pressed;
        }

        // =========================================================
        // 2. 데드맨 (Button 1) — 누르고 있는 동안에만 teleop
        // =========================================================
        // 이전에는 반대였다: 누르면 멈추는 클러치. 그러면 아무것도 안 누른 상태가
        // "계속 움직임"이라, 손을 놓거나 스타일러스를 떨어뜨리면 로봇이 계속 간다.
        // 데드맨으로 뒤집으면 놓는 순간 정지한다.
        //
        // 정지 시에도 0 twist 를 계속 발행한다. 발행을 멈추면 us_diff_ik_node 의
        // 100 ms twist 워치독이 걸려 프로브 -z 후퇴가 시작된다 (§12.2). 여기서
        // 원하는 것은 후퇴가 아니라 정지다.
        bool is_btn1_pressed = (s.buttons & HD_DEVICE_BUTTON_1) != 0;

        if (!is_btn1_pressed) {
            ctx->twist_pub->publish(geometry_msgs::msg::Twist());
            s.filtered_lin_vel.setZero();
            s.filtered_ang_vel.setZero();
            s.prev_position = s.position;   // 누르는 순간 튀지 않도록 차분 기준 갱신
            s.prev_rotation = s.rotation;
            s.last_time = now;
            if (s.was_deadman_engaged) {
                RCLCPP_INFO(this->get_logger(), "[%s] 데드맨 해제 — 정지", ctx->prefix.c_str());
                s.was_deadman_engaged = false;
            }
            return;
        }
        if (!s.was_deadman_engaged) {
            RCLCPP_INFO(this->get_logger(), "[%s] 데드맨 engage — teleop 활성", ctx->prefix.c_str());
            s.was_deadman_engaged = true;
        }

        // =========================================================
        // 3. TWIST CALCULATION
        // =========================================================
        double dt = (now - s.last_time).seconds();
        if (dt <= 0.0001) return;

        // Raw Velocity (World Frame)
        Eigen::Vector3d vel_world = (s.position - s.prev_position) / dt;

        Eigen::Matrix3d rot_diff = s.rotation * s.prev_rotation.transpose();
        Eigen::AngleAxisd aa(rot_diff);
        Eigen::Vector3d ang_vel_world = aa.axis() * aa.angle() / dt;

        // Project to Body Frame
        Eigen::Vector3d vel_body = s.rotation.transpose() * vel_world;
        Eigen::Vector3d ang_vel_body = s.rotation.transpose() * ang_vel_world;

        // Filter
        s.filtered_lin_vel = alpha * vel_body + (1.0 - alpha) * s.filtered_lin_vel;
        s.filtered_ang_vel = alpha * ang_vel_body + (1.0 - alpha) * s.filtered_ang_vel;

        // Deadzone — 병진·회전 모두 프로파일 값을 쓴다.
        // 회전 쪽은 예전에 0.1 rad/s 로 코드에 박혀 있었다. 그 값이면 RUS 각속도
        // 상한 0.2 rad/s 에 대해 최소 지령이 0.06 rad/s 라 사용 구간이 3:1 뿐이었다.
        if (s.filtered_lin_vel.norm() < profile.linear_deadzone) s.filtered_lin_vel.setZero();
        if (s.filtered_ang_vel.norm() < profile.angular_deadzone) s.filtered_ang_vel.setZero();

        // ---- 축 매핑 (초음파 프로브 기준, 2026-08-18 실측으로 확정) ----
        //
        // 이전 값은 복강경 툴 기준이었고 "Signs fixed as per original code" 라는
        // 주석만 달린 채 근거가 남아 있지 않았다. 프로브로 바꾸면서 다시 측정했다.
        //
        // 측정 (구 매핑 기준, 손 동작 → 발행된 축):
        //     오른쪽으로 밀기 → -lin y      (2위 대비 3.0배)
        //     앞쪽으로  밀기 → +lin z      (1.4배 — 분리 약함)
        //     아래로   누르기 → +lin x      (6.5배)
        //   세 축을 행렬로 놓으면 행렬식이 +1 이다. 즉 하나의 강체 회전으로 설명되며,
        //   2번의 분리비가 낮아도 나머지 두 축과 기하학적으로 모순되지 않는다.
        //
        // 목표 (DESIGN_NOTES §4.1 프로브 프레임):
        //     +z = 조직 침투(법선), +x = lateral(영상면 내), +y = elevational
        //   → 아래로 누르기 = +z,  오른쪽 = +x,  앞쪽 = -y
        //
        //   앞쪽이 -y 인 것은 임의 선택이 아니다. 오른쪽=+x, 아래=+z 를 고정하면
        //   남은 부호는 행렬식이 +1 이어야 한다는 조건으로 결정된다. +y 로 두면
        //   행렬식이 -1 이 되어 회전이 아니라 거울상이 된다.
        //
        // 구 → 신 변환을 풀면 x 는 그대로, y 와 z 는 부호 반전이다
        // (장치 x 축 기준 180° 회전 하나).
        //
        // 회전축에도 **같은** 회전을 적용한다. 병진과 회전에 다른 변환을 쓰면
        // 그 twist 는 강체 운동이 아니게 된다. 실측 회전 부호를 그대로 쓰지 않은
        // 이유는 측정 안내가 "기울이세요/비트세요" 라는 왕복 동작이어서 피크 부호가
        // 어느 방향이 더 빨랐는지에 좌우되기 때문이다. 축 대응(부호 무시)은
        // pitch→rx, roll→ry, yaw→rz 로 실측과 일치하며, yaw→rz 는 §5 에서
        // policy 가 담당하는 영상면 회전축과 맞는다.
        //
        // ⏳ 회전 3축의 부호는 방향을 지정한 재측정으로 확인해야 한다.
        geometry_msgs::msg::Twist twist;

        // ---- 스타일러스 ≡ 프로브 (규칙 기반, 2026-08-18) ----
        //
        // 이전에는 손동작 속도를 재서 축을 맞췄다. 그 방식은 손목 결합과 파지 각도에
        // 오염된다 — 5 회 반복 측정에서도 "앞쪽"이 "오른쪽"·"아래"와 각각 71° 로 나와
        // 측정 행렬의 행렬식이 -0.885 였다. 회전을 정의할 수 없는 데이터다.
        //
        // 대신 두 프레임의 **문서화된 규약**으로 정한다.
        //
        // OpenHaptics 장치/스타일러스 프레임 (OpenHaptics_ProgGuide):
        //   +X 오른쪽, +Y 위, +Z 화면 밖 = 조작자 쪽
        //   "the pencil should be designed to lie along the Z-axis with the
        //    pencil tip at the origin"  → 촉이 가리키는 방향은 -Z
        //
        // 프로브 프레임 (DESIGN_NOTES §4.1):
        //   +z 조직 침투 방향(= 프로브가 가리키는 방향), +x lateral, +y elevational
        //
        // 동일시:
        //   스타일러스 -Z (촉 방향) ≡ 프로브 +z (침투 방향)
        //   스타일러스 +X (오른쪽)  ≡ 프로브 +x (lateral)
        //   → 남은 축은 det=+1 조건이 결정한다:  A = diag(+1, -1, -1)
        //
        // 병진과 회전에 **같은** A 를 쓴다. 다르게 쓰면 twist 가 강체 운동이 아니게 된다.
        // 속도는 이미 스타일러스 body 프레임으로 투영되어 있으므로(위 참조),
        // A 를 곱하면 그대로 프로브 프레임 twist 가 된다.
        //
        // ⏳ 침투축 대응은 규약으로 확정됐지만, **그 축을 중심으로 한 회전(roll)** 은
        //    아직 자유도가 남아 있다. 프로브가 실제로 장착되어 영상면이 플랜지에 대해
        //    어떻게 놓이는지 확인되면 확정된다. 그때까지 +X ≡ +x 는 잠정 규약이다.
        //   A0 = diag(+1,-1,-1)                      촉 방향 ≡ 침투 방향
        //   A  = Rz(tip_roll_deg) * A0                침투축 둘레 남은 자유도
        //
        // 회전으로 구현하는 이유: "pitch 와 roll 을 맞바꾼다" 를 단순 축 교환으로
        // 하면 행렬식이 -1 이 되어 회전이 아니라 반사가 된다. Rz 를 곱하면
        // det=+1 이 구조적으로 보장된다.
        //
        // 병진과 회전에 같은 A 를 쓴다. 회전만 바꾸면 강체 운동이 아니게 된다.
        const double th = profile.tip_roll_deg * M_PI / 180.0;
        const double c = std::cos(th), sn = std::sin(th);
        Eigen::Matrix3d A0 = Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal();
        Eigen::Matrix3d Rz;
        Rz <<  c, -sn, 0.0,
              sn,   c, 0.0,
             0.0, 0.0, 1.0;
        const Eigen::Matrix3d A = Rz * A0;

        // ---- 축별 병진 이득 (조작자 손 기준, 2026-08-31) ----
        //
        // 조작 피드백: 앞뒤·상하 병진이 손 움직임에 비해 답답하다. 좌우는 그대로 두고
        // 두 축만 키우려면 이득이 **장치 world 프레임**에 걸려야 한다
        // (+X 오른쪽, +Y 위, +Z 조작자 쪽 → 상하는 y, 앞뒤는 z).
        //
        // filtered_lin_vel 은 스타일러스 body 프레임이다. 거기에 그냥 곱하면 이득 축이
        // 손목을 따라 돌아, 같은 "앞으로 밀기" 가 파지 각도마다 다르게 증폭된다.
        // 그래서 world 로 되돌려 곱하고 body 로 돌아온다:
        //     v_body' = Rᵀ · diag(gain) · R · v_body
        //
        // 하류(us_diff_ik, teleop.linear_frame: latched)는 R_stylus·Aᵀ 로 지령을 장치
        // world 로 되돌린 뒤 고정 행렬 하나로 base 에 싣는다 (§10.5). 그 경로를 지나면
        // 이득은 정확히 diag(gain)·v_world 로 남는다 — 스타일러스 자세와 무관하다.
        //
        // 데드존 **뒤**에 곱한다. 앞에 곱하면 문턱이 축마다 달라져, 이득을 올린 축의
        // 손떨림이 그만큼 더 통과한다 (데드존 값은 손 잡음 실측이다).
        //
        // 방향이 아니라 축에 걸린다 — 앞으로 밀 때와 뒤로 뺄 때의 배율이 같다.
        // 다르면 왔던 경로를 같은 손동작으로 되짚을 수 없다.
        const Eigen::Vector3d gain(profile.lin_gain[0], profile.lin_gain[1], profile.lin_gain[2]);
        Eigen::Vector3d lin_body = s.filtered_lin_vel;
        if (gain != Eigen::Vector3d::Ones()) {
            lin_body = s.rotation.transpose() * gain.cwiseProduct(s.rotation * lin_body);
        }

        const Eigen::Vector3d v = A * lin_body * l_scale;
        const Eigen::Vector3d w = A * s.filtered_ang_vel * a_scale;

        twist.linear.x  = profile.lin_sign[0] * v.x();
        twist.linear.y  = profile.lin_sign[1] * v.y();
        twist.linear.z  = profile.lin_sign[2] * v.z();

        twist.angular.x = profile.ang_sign[0] * w.x();
        twist.angular.y = profile.ang_sign[1] * w.y();
        twist.angular.z = profile.ang_sign[2] * w.z();

        // 스타일러스 자세를 그대로 발행한다. 위 규약이 실제 장치와 맞는지 확인하려면
        // 속도가 아니라 **자세**를 봐야 한다 — 자세는 장치가 직접 주는 값이라
        // 손목 결합에 오염되지 않는다.
        {
            geometry_msgs::msg::PoseStamped ps;
            ps.header.stamp = now;
            ps.header.frame_id = "touch_device";
            ps.pose.position.x = s.position.x();
            ps.pose.position.y = s.position.y();
            ps.pose.position.z = s.position.z();
            const Eigen::Quaterniond quat(s.rotation);
            ps.pose.orientation.w = quat.w();
            ps.pose.orientation.x = quat.x();
            ps.pose.orientation.y = quat.y();
            ps.pose.orientation.z = quat.z();
            ctx->stylus_pub->publish(ps);
        }

        ctx->twist_pub->publish(twist);

        s.prev_position = s.position;
        s.prev_rotation = s.rotation;
        s.last_time = now;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    // 장치를 못 여는 것은 흔한 상황이다 — TouchCheckup 이 떠 있거나, 이 노드가
    // 이미 하나 돌고 있으면 점유 실패한다. 예외를 그대로 두면 terminate 로
    // SIGABRT + 코어덤프가 나고 데스크톱에 "system error" 팝업이 뜬다.
    // 진단에 도움이 안 되므로 메시지를 내고 조용히 실패한다.
    std::shared_ptr<TouchTeleopNode> node;
    try {
        node = std::make_shared<TouchTeleopNode>();
    } catch (const std::exception &e) {
        RCLCPP_ERROR(rclcpp::get_logger("touch_twist"),
                     "기동 실패: %s\n"
                     "  · 다른 touch_twist 나 TouchCheckup 이 장치를 잡고 있는지 확인\n"
                     "  · lsusb 에 2988:0304 가 보이는지 확인\n"
                     "  · Touch_HeadlessSetup name=\"Default Device\" dev=HID 로 설정 재생성",
                     e.what());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}