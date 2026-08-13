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

    TouchTeleopNode() : Node("touch_teleop_node") {
        // --- Parameters ---
        this->declare_parameter<double>("linear_scale", 0.6);
        this->declare_parameter<double>("angular_scale", 0.6);
        this->declare_parameter<double>("filter_alpha", 0.4); 
        this->declare_parameter<double>("deadzone", 0.001); 

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
            
            RCLCPP_INFO(this->get_logger(), "Touch Teleop Ready.");
            RCLCPP_INFO(this->get_logger(), " - Button 1 (Gray): Hold to PAUSE (Clutch)");
            RCLCPP_INFO(this->get_logger(), " - Button 2 (White): Hold to GRIP (30), Release to OPEN (100)");
            RCLCPP_INFO(this->get_logger(), "Active Devices: %zu", devices_.size());
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
        std::string twist_topic = "/fr5_" + side + "/desired_twist";
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

    // --- ROS Loop (50Hz) ---
    void timerCallback() {
        rclcpp::Time now = this->now();

        double l_scale = this->get_parameter("linear_scale").as_double();
        double angular_scale_param = this->get_parameter("angular_scale").as_double();
        double alpha   = this->get_parameter("filter_alpha").as_double();
        double deadzone = this->get_parameter("deadzone").as_double();

        std::lock_guard<std::mutex> lock(state_mutex_);

        for (auto &ctx : devices_) {
            if (!ctx->state.valid) continue;

            processDevice(ctx, now, l_scale, angular_scale_param, alpha, deadzone);
        }
    }

    void processDevice(std::shared_ptr<DeviceContext> ctx, rclcpp::Time now, 
                       double l_scale, double a_scale, double alpha, double deadzone) 
    {
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
        if (is_btn2_pressed) {
            gripper_msg.data = 5.0;  // Pressed -> Close/Grip
        } else {
            gripper_msg.data = 100.0; // Released -> Open
        }
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

        // Deadzone
        if (s.filtered_lin_vel.norm() < deadzone) s.filtered_lin_vel.setZero();
        if (s.filtered_ang_vel.norm() < 0.1) s.filtered_ang_vel.setZero();

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