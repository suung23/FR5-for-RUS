#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <HD/hd.h>
#include <HDU/hduVector.h>
#include <HDU/hduMatrix.h>
#include <HDU/hduError.h>

#include <mutex>
#include <memory>
#include <chrono>
#include <functional>
#include <cmath>
#include <vector>
#include <string>

class TouchTeleopNode : public rclcpp::Node {   
public: 
    struct TouchState {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Transformation matrix (column-major)
        Eigen::Matrix4d transform;
        bool button1_clicked = false;
        bool button2_clicked = false;
        bool valid = false;
    };

    struct DeviceContext {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::string name;
        HHD hHD;
        rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_pub;
        TouchState state;
        Eigen::Matrix4d T_zero;
        bool is_zero_set = false;
        
        DeviceContext() {
            T_zero.setIdentity();
        }
    };

    TouchTeleopNode() : Node("touch_teleop_node") {
        this->declare_parameter<double>("linear_scale", 1.0);
        this->declare_parameter<double>("angular_scale", 1.0);
        
        // Declare device names params with default values
        this->declare_parameter<std::string>("left_dev_name", "touch_left");
        this->declare_parameter<std::string>("right_dev_name", "Touch_Right");

        std::string left_name = this->get_parameter("left_dev_name").as_string();
        std::string right_name = this->get_parameter("right_dev_name").as_string();

        // Not sure about this...
        R_S_D_ << 0, 0, -1, 
                -1, 0, 0, 
                0, 1, 0;

        // Initialize Haptic Devices
        // Try to initialize Left
        initDevice(left_name, "touch/left/joy");
        // Try to initialize Right
        initDevice(right_name, "touch/right/joy");

        if (devices_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No haptic devices initialized. Please check device names.");
            // Raise a system exit or just let it fall through? 
            // Since we are in the constructor, we can't easily stop 'spin' from here unless we throw.
            // But better pattern:
            throw std::runtime_error("No haptic devices found");
        }

        // Start Haptic Scheduler
        hd_scheduler_handle_ = hdScheduleAsynchronous(touchCallback, this, HD_MAX_SCHEDULER_PRIORITY);
        hdStartScheduler();

        timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&TouchTeleopNode::timerCallback, this));
        RCLCPP_INFO(this->get_logger(), "Touch Teleop Node Started. %zu devices active.", devices_.size());
    }

    ~TouchTeleopNode() {
        hdStopScheduler();
        hdUnschedule(hd_scheduler_handle_);
        // Clean up devices if needed (hdDisableDevice is usually not strictly required if exiting)
    }

private: 
    std::vector<std::shared_ptr<DeviceContext>> devices_;
    std::mutex state_mutex_;
    Eigen::Matrix3d R_S_D_; // Rotation matrix from Device to Robot

    HDSchedulerHandle hd_scheduler_handle_;
    rclcpp::TimerBase::SharedPtr timer_;

    void initDevice(const std::string& dev_name, const std::string& topic_name) {
        HDErrorInfo error;
        HHD hHD = hdInitDevice(dev_name.c_str());
        
        if (HD_DEVICE_ERROR(error = hdGetError())) {
            // Log warning but don't fail completely if one is missing, unless needed.
            // But usually we want to know.
            RCLCPP_WARN(this->get_logger(), "Failed to initialize haptic device '%s'. Error: %s", dev_name.c_str(), hdGetErrorString(error.errorCode));
            // Try fallback if name is "Default Device"? No, explicitly asked for multi support.
            return;
        }

        auto ctx = std::make_shared<DeviceContext>();
        ctx->name = dev_name;
        ctx->hHD = hHD;
        ctx->joy_pub = this->create_publisher<sensor_msgs::msg::Joy>(topic_name, 10);
        
        devices_.push_back(ctx);
        RCLCPP_INFO(this->get_logger(), "Initialized device '%s' publishing to '%s'", dev_name.c_str(), topic_name.c_str());
    }

    static HDCallbackCode HDCALLBACK touchCallback(void *pUserData) {
        TouchTeleopNode *node = static_cast<TouchTeleopNode *>(pUserData);

        for (auto &ctx : node->devices_) {
            hdMakeCurrentDevice(ctx->hHD);
            hdBeginFrame(ctx->hHD);

            double raw_transform[16] = {0};
            hdGetDoublev(HD_CURRENT_TRANSFORM, raw_transform);

            HDint buttons = 0;
            hdGetIntegerv(HD_CURRENT_BUTTONS, &buttons);

            hdEndFrame(ctx->hHD);

            // Update state (Mutex protected)
            {
                std::lock_guard<std::mutex> lock(node->state_mutex_);
                ctx->state.transform = Eigen::Map<Eigen::Matrix4d>(raw_transform);
                
                // Convert mm -> m (units of position)
                ctx->state.transform.block<3, 1>(0, 3) *= 0.001;

                ctx->state.button1_clicked = (buttons & HD_DEVICE_BUTTON_1) != 0;
                ctx->state.button2_clicked = (buttons & HD_DEVICE_BUTTON_2) != 0;
                ctx->state.valid = true;
            }
        }
        
        return HD_CALLBACK_CONTINUE;
    }

    void timerCallback() {
        double linear_scale = this->get_parameter("linear_scale").as_double();
        double angular_scale = this->get_parameter("angular_scale").as_double();

        std::lock_guard<std::mutex> lock(state_mutex_);

        for (auto &ctx : devices_) {
             processDevice(ctx, linear_scale, angular_scale);
        }
    }

    void processDevice(std::shared_ptr<DeviceContext> ctx, double linear_scale, double angular_scale) {
        // Do nothing if data is not valid
        if (!ctx->state.valid) {
            return;
        }

        // If zero point is not set, set it
        if (!ctx->is_zero_set) {
            // [Optional] Wait if value is too large or 0
            if (ctx->state.transform.isIdentity(0.001)) return;

            ctx->T_zero = ctx->state.transform; // Set current position as zero point
            ctx->is_zero_set = true;               // Set zero point flag
            
            RCLCPP_INFO(this->get_logger(), "Zero Point Set for device %s. Ready to move.", ctx->name.c_str());
            return;
        }

        // Linear Velocity
        Eigen::Vector3d p_current = ctx->state.transform.block<3, 1>(0, 3);
        Eigen::Vector3d p_zero = ctx->T_zero.block<3, 1>(0, 3);
        Eigen::Vector3d p_diff = p_current - p_zero;

        // Angular Velocity
        Eigen::Matrix3d R_curr = ctx->state.transform.block<3, 3>(0, 0);
        Eigen::Matrix3d R_zero = ctx->T_zero.block<3, 3>(0, 0);
        Eigen::Matrix3d R_diff = R_curr * R_zero.transpose();

        Eigen::AngleAxisd angle_axis(R_diff);
        Eigen::Vector3d w_diff = angle_axis.axis() * angle_axis.angle();

        // Convert to robot frame
        Eigen::Vector3d v_robot = R_S_D_ * p_diff * linear_scale;
        Eigen::Vector3d w_robot = R_S_D_ * w_diff * angular_scale;

        sensor_msgs::msg::Joy joy_msg;
        
        // header
        joy_msg.header.stamp = this->now();
        joy_msg.header.frame_id = ctx->name; // Use device name as frame_id

        // twist (linear and angular)
        joy_msg.axes.resize(6);
        joy_msg.axes[0] = v_robot.x();
        joy_msg.axes[1] = v_robot.y();
        joy_msg.axes[2] = v_robot.z();
        joy_msg.axes[3] = w_robot.x();
        joy_msg.axes[4] = w_robot.y();
        joy_msg.axes[5] = w_robot.z();

        // buttons
        joy_msg.buttons.resize(2);
        joy_msg.buttons[0] = ctx->state.button1_clicked ? 1 : 0;
        joy_msg.buttons[1] = ctx->state.button2_clicked ? 1 : 0;

        ctx->joy_pub->publish(joy_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TouchTeleopNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 