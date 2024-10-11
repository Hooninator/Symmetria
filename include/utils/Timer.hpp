#ifndef TIMER_HPP
#define TIMER_HPP

#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <mpi.h>

namespace symmetria {

class Timer {
private:
    struct TimerData {
        double start_time = 0.0;
        double elapsed_time = 0.0;
        bool running = false;
    };

    std::unordered_map<std::string, TimerData> timers;

    // Use MPI_Wtime() to get the current time
    double get_current_time() {
        return MPI_Wtime();
    }

public:

    Timer() {timers.reserve(20);}

    // Start a timer with the given name
    void start_timer(const std::string& name) {
        TimerData& timer = timers[name];
        if (!timer.running) {
            timer.start_time = get_current_time();
            timer.running = true;
        }
    }

    // Stop the timer with the given name
    void stop_timer(const std::string& name) {
        auto it = timers.find(name);
        if (it != timers.end()) {
            TimerData& timer = it->second;
            if (timer.running) {
                double end_time = get_current_time();
                timer.elapsed_time += end_time - timer.start_time;
                timer.running = false;
            } else {
                std::cerr << "Timer \"" << name << "\" is not running." << std::endl;
            }
        } else {
            std::cerr << "Timer \"" << name << "\" not found." << std::endl;
        }
    }

    // Get the elapsed time for the timer with the given name
    double get_timer(const std::string& name) {
        auto it = timers.find(name);
        if (it != timers.end()) {
            TimerData& timer = it->second;
            if (timer.running) {
                double current_time = get_current_time();
                return timer.elapsed_time + (current_time - timer.start_time);
            } else {
                return timer.elapsed_time;
            }
        } else {
            std::cerr << "Timer \"" << name << "\" not found." << std::endl;
            return 0.0;
        }
    }

    // Write the output of a single timer to a JSON file
    void write_timer(const std::string& name, const std::string& filename) {
        auto it = timers.find(name);
        if (it != timers.end()) {
            double time = get_timer(name);
            std::ofstream file(filename);
            if (file.is_open()) {
                file << "{\n";
                file << "  \"" << name << "\": " << time << "\n";
                file << "}\n";
                file.close();
            } else {
                std::cerr << "Unable to open file: " << filename << std::endl;
            }
        } else {
            std::cerr << "Timer \"" << name << "\" not found." << std::endl;
        }
    }

    // Write the outputs of all timers to a JSON file
    void write_all_timers(const std::string& filename, 
                          const std::ios_base::openmode& mode) {
		std::ofstream file(filename, mode);
        if (file.is_open()) {
            // Write the header
            if (mode==std::ios_base::trunc)
                file << "Timer Name,Elapsed Time (s)\n";
            for (const auto& pair : timers) {
                const std::string& name = pair.first;
                double time = get_timer(name);
                file << name << "," << time << "\n";
            }
            file.close();
        } else {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }
    }


    void clear_all_timers()
    {
        timers.clear();
    }


    void clear_one_timer(const std::string& name)
    {
        timers.erase(name);
    }
};
}
#endif
