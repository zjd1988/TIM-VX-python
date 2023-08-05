/********************************************
// Filename: timvx_log.cpp
// Created by zhaojiadi on 2021/5/28
// Description: 
// 
********************************************/
#include "common/timvx_log.h"
#include "spdlog/sinks/rotating_file_sink.h"

namespace TimVX 
{

    TimVXLog& TimVXLog::Instance()
    {
        static TimVXLog log;
        return log;
    }

    void TimVXLog::initTimVXLog(std::string logger_name, std::string file_name, int log_level)
    {
        // set log level
        if (log_level != TIMVX_LEVEL_TRACE && log_level != TIMVX_LEVEL_DEBUG && 
            log_level != TIMVX_LEVEL_INFO  && log_level != TIMVX_LEVEL_WARN  && 
            log_level != TIMVX_LEVEL_ERROR && log_level != TIMVX_LEVEL_FATAL && 
            log_level != TIMVX_LEVEL_OFF)
            log_level = TIMVX_LEVEL_INFO;

        spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));

        // set log rotate
        int log_file_size = TIMVX_LOG_FILE_SIZE;
        if (log_file_size <= 0)
            log_file_size = 50 * 1024 * 1024; // 50MB
        int num_log_files = TIMVX_LOG_FILE_NUM;
            num_log_files = 3;
        if (file_name.empty())
            file_name = "server_log.txt";
        if (logger_name.empty())
            logger_name = "server_log";

        std::cout << "log level: " << log_level << std::endl;
        std::cout << "logger name: " << logger_name << std::endl;
        std::cout << "log file path: " << file_name << std::endl;
        auto console_sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(file_name, log_file_size, num_log_files);
        m_logger.reset();
        m_logger = std::unique_ptr<spdlog::logger>(new spdlog::logger(logger_name, {console_sink, file_sink}));

        // set log pattern
        m_logger->flush_on(spdlog::level::info);
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %z] [%n] [%^---%L---%$] [thread %t] [%s %! %#] %v");
        // m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %z] [%n] [%^---%L---%$] [thread %t] %v");
        spdlog::set_default_logger(m_logger);
    }

    void TimVXLog::stopTimVXLog()
    {
        spdlog::shutdown();
    }

    TimVXLog::~TimVXLog()
    {
        stopTimVXLog();
    }

    void TimVXLog::setLevel(int level)
    {
        spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
    }

} // namespace TimVX