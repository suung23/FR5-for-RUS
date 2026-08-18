#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
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
    };

    struct DeviceContext {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::string name;
        HHD hHD;
        
        // Each device has its own publishers
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gripper_pub;

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

        // 초음파 스택에서는 조작자 twist 가 admittance 를 거쳐 나간다. admittance 가
        // 유일한 desired_twist 발행자여야 둘이 다투지 않고, TELEOP 모드에서 6축 통과가
        // 성립한다. 빈 문자열이면 기존 동작(팔별 desired_twist 직접 발행).
        this->declare_parameter<std::string>("teleop.twist_topic_override", "");

        this->declare_parameter<std::string>("left_dev_name", "touch_left");
        this->declare_parameter<std::string>("right_dev_name", "Touch_Right");

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
            RCLCPP_INFO(this->get_logger(), " - Button 1 (Gray): Hold to PAUSE (Clutch)");
            RCLCPP_INFO(this->get_logger(), " - Button 2 (White): Hold to GRIP (%.0f), Release to OPEN (%.0f)",
                        GRIPPER_CLOSED, GRIPPER_OPEN);
            RCLCPP_INFO(this->get_logger(), "Active Devices: %zu", devices_.size());
            RCLCPP_INFO(this->get_logger(),
                "Profile '%s': lin x%.3f (deadzone %.4f m/s), ang x%.3f (deadzone %.4f rad/s), alpha %.2f",
                this->get_parameter("teleop.profile").as_string().c_str(),
                p.linear_scale, p.linear_deadzone, p.angular_scale, p.angular_deadzone, p.filter_alpha);
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

    void initDevice(const std::string& dev_name, const std::string& side) {
        HDErrorInfo error;
        HHD hHD = hdInitDevice(dev_name.c_str());
        
        if (hHD == HD_INVALID_HANDLE || HD_DEVICE_ERROR(error = hdGetError())) {
            RCLCPP_WARN(this->get_logger(), "Failed to init device '%s' (%s): %s", dev_name.c_str(), side.c_str(), hdGetErrorString(error.errorCode));
            return;
        }

        auto ctx = std::make_shared<DeviceContext>();
        ctx->name = dev_name;
        ctx->hHD = hHD;
        ctx->prefix = "touch/" + side; // e.g., touch/left

        // Create publishers with specific topics
        const std::string override_topic =
            this->get_parameter("teleop.twist_topic_override").as_string();
        std::string twist_topic = override_topic.empty()
            ? "/fr5_" + side + "/desired_twist"
            : override_topic;
        std::string gripper_topic = "/fr5_" + side + "/desired_gripper_pose";

        ctx->twist_pub = this->create_publisher<geometry_msgs::msg::Twist>(twist_topic, 10);
        ctx->gripper_pub = this->create_publisher<std_msgs::msg::Float32>(gripper_topic, 10);

        // Init state time
        ctx->state.last_time = this->now();

        devices_.push_back(ctx);
        RCLCPP_INFO(this->get_logger(), "Initialized %s on topics: %s, %s", dev_name.c_str(), twist_topic.c_str(), gripper_topic.c_str());
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
        if (name == "laparoscopic" || name == "us_approach") {
            key = name;
        } else {
            RCLCPP_ERROR(this->get_logger(),
                "알 수 없는 teleop.profile '%s'. 가장 보수적인 us_approach 로 대체한다. "
                "'laparoscopic' 또는 'us_approach' 여야 한다.", name.c_str());
            key = "us_approach";
        }
        const std::string p = "teleop." + key + ".";
        return Profile{
            this->get_parameter(p + "linear_scale").as_double(),
            this->get_parameter(p + "angular_scale").as_double(),
            this->get_parameter(p + "filter_alpha").as_double(),
            this->get_parameter(p + "linear_deadzone").as_double(),
            this->get_parameter(p + "angular_deadzone").as_double(),
        };
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
        // 2. CLUTCH LOGIC (Button 1)
        // =========================================================
        bool is_btn1_pressed = (s.buttons & HD_DEVICE_BUTTON_1) != 0;

        if (is_btn1_pressed) {
            ctx->twist_pub->publish(geometry_msgs::msg::Twist()); // Stop robot
            s.filtered_lin_vel.setZero();
            s.filtered_ang_vel.setZero();
            s.prev_position = s.position; // Reset diff
            s.prev_rotation = s.rotation;
            s.last_time = now;
            return;
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

        // Mapping & Publish
        geometry_msgs::msg::Twist twist;
        
        // Axis Mapping (Signs fixed as per original code: +X/+Y/+Z -> +Z/+X/+Y from OH Frame)
        twist.linear.x =  -(s.filtered_lin_vel.z() * l_scale);
        twist.linear.y =  -( s.filtered_lin_vel.x() * l_scale); 
        twist.linear.z =  (s.filtered_lin_vel.y() * l_scale);

        twist.angular.x =  -(s.filtered_ang_vel.z() * a_scale); 
        twist.angular.y =  -(s.filtered_ang_vel.x() * a_scale);
        twist.angular.z =  s.filtered_ang_vel.y() * a_scale;

        ctx->twist_pub->publish(twist);

        s.prev_position = s.position;
        s.prev_rotation = s.rotation;
        s.last_time = now;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TouchTeleopNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}