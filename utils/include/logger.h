
#include <iostream>
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>

#define LOG_INFO(message) Logger::getInstance().log("INFO", __FILE__, __LINE__, message)
#define LOG_ERROR(message) Logger::getInstance().log("ERROR", __FILE__, __LINE__, message)
#define LOG_WARNING(message) Logger::getInstance().log("WARNING", __FILE__, __LINE__, message)
#define LOG_DEBUG(message) Logger::getInstance().log("DEBUG", __FILE__, __LINE__, message)

class Logger{
    public:
        static Logger& getInstance() {
            static Logger instance;
            return instance;
        }
        ~Logger() {
            if (log_file.is_open()) {
                log_file.close();
            }
        }
        void log(
            const std::string& level, 
            const char* file,
            size_t line, 
            const std::string& message
        ) {
            std::lock_guard<std::mutex> lock(log_mutex);

            auto now = std::chrono::system_clock::now();
            std::time_t now_time = std::chrono::system_clock::to_time_t(now);
            std::tm tm_now;
            localtime_r(&now_time, &tm_now);
            std::ostringstream oss;
            oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
            std::string timestamp = oss.str();

            std::cout << "[" << timestamp << "] [" << level << "] " << file << ":" << line << " - " << message << std::endl;

            if (log_file.is_open()) {
                log_file << "[" << timestamp << "] [" << level << "] " << file << ":" << line << " - " << message << std::endl;
            }
        }


    private:
        Logger(bool to_file = false) {
            if (to_file) {
                log_file.open("log.txt", std::ios::out | std::ios::app);
            }
        };
        static Logger* instance;
        std::ofstream log_file;
        std::mutex log_mutex;
}