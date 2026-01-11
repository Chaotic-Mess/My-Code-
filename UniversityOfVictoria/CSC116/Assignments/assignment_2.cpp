/*
 * Program: BC Ferries Data Analysis (assignment_2)
 * Author: Zac Matthias
 * Date: Oct 19th 2025
 * Description:
 *   Parses BC Ferries sailing data and analyzes on-time performance.
 *   Tracks route statistics and identifies best/worst days based on late sailings.
 *
 * Libraries used:
 *   - <iostream> for input/output
 *   - <vector> for dynamic arrays
 *   - <string> for string manipulation
 *   - <cctype> for character handling
 *   - <fstream> for file operations
 *   - <stdexcept> for exceptions
 *   - <iomanip> for formatting
 *
 *   PS E:\CSC116> g++ assignment_2.cpp -o assignment_2
 *   PS E:\CSC116> ./assignment_2 action input_filename
 *
 *  where action is either 'route_summary' or 'days'
 *  Example:
 *   g++ assignment_2.cpp -o assignment_2
 *  ./assignment_2 days Data/Assn2/11_ShortLineI.txt
 *
 */
#include <vector>    // Used in functions: split_csv, compute_day_stats, performance_by_route, best_days, worst_days
#include <iostream>  // Used in functions: read_sailings, print_sailing, main
#include <string>    // Used in functions: trim, is_whitespace_only, begins_with_digit, split_csv, parse_sailing, read_sailings, print_sailing, main
#include <cctype>    // Used in functions: trim, is_whitespace_only, begins_with_digit
#include <fstream>   // Used in functions: read_sailings
#include <stdexcept> // Used in functions: read_sailings
#include <iomanip>   // Used in functions: print_sailing

// Date structure
struct Date
{
    int day{0};
    int month{0};
    int year{0};
};

// Time structure
struct TimeOfDay
{
    int hour{0};
    int minute{0};
};

// Single sailing record
struct Sailing
{
    int route_number{0};
    std::string source_terminal{""};
    std::string dest_terminal{""};
    std::string vessel_name{""};

    Date departure_date{};
    TimeOfDay scheduled_departure_time{};

    int expected_duration{0};
    int actual_duration{0};
};

// Stats for a route
struct RouteStatistics
{
    int route_number{0};
    int total_sailings{0};
    int late_sailings{0};
};

// Stats for a day
struct DayStatistics
{
    Date date{};
    int total_sailings{0};
    int late_sailings{0};
};

// Parsing exceptions
struct IncompleteLineException
{
    unsigned int num_fields{};
};

struct EmptyFieldException
{
    unsigned int which_field{};
};

struct NonNumericDataException
{
    std::string bad_field{};
};

struct InvalidTimeException
{
    TimeOfDay bad_time{};
};

// Helper functions

// Remove leading/trailing spaces
static std::string trim(std::string s)
{
    size_t i = 0, j = s.size();
    while (i < j && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1])))
        --j;
    return s.substr(i, j - i);
}

// Check if string is empty or only whitespace
static bool is_whitespace_only(std::string const &s)
{
    for (unsigned char c : s)
        if (!std::isspace(c))
            return false;
    return true;
}

// Check if string starts with a digit
static bool begins_with_digit(std::string const &s)
{
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    return (i < s.size()) && std::isdigit(static_cast<unsigned char>(s[i]));
}

// Split CSV line by commas
static std::vector<std::string> split_csv(std::string const &line)
{
    std::vector<std::string> out;
    std::string cur;
    out.reserve(16);
    for (char ch : line)
    {
        if (ch == ',')
        {
            out.push_back(cur);
            cur.clear();
        }
        else
        {
            cur.push_back(ch);
        }
    }
    out.push_back(cur);
    return out;
}

// Check if sailing is late (5+ minutes over expected)
static inline bool is_late(int actual, int expected)
{
    return actual >= expected + 5;
}

// Compare two dates for equality
static bool dates_equal(Date const &a, Date const &b)
{
    return a.year == b.year && a.month == b.month && a.day == b.day;
}

// Group sailings by day and compute stats
static std::vector<DayStatistics> compute_day_stats(const std::vector<Sailing> &sailings)
{
    std::vector<DayStatistics> days;

    for (auto const &s : sailings)
    {
        // Find if this date already exists
        bool found = false;
        for (auto &d : days)
        {
            if (dates_equal(d.date, s.departure_date))
            {
                d.total_sailings += 1;
                if (is_late(s.actual_duration, s.expected_duration))
                {
                    d.late_sailings += 1;
                }
                found = true;
                break;
            }
        }

        // If date not found, add new entry
        if (!found)
        {
            DayStatistics new_day;
            new_day.date = s.departure_date;
            new_day.total_sailings = 1;
            new_day.late_sailings = is_late(s.actual_duration, s.expected_duration) ? 1 : 0;
            days.push_back(new_day);
        }
    }

    return days;
}

// Parse a CSV line into a Sailing object
Sailing parse_sailing(std::string const &input_line)
{
    auto fields = split_csv(input_line);

    // Error check 1: Must have exactly 11 fields
    if (fields.size() != 11U)
    {
        IncompleteLineException e;
        e.num_fields = static_cast<unsigned int>(fields.size());
        throw e;
    }

    // Error check 2: No field can be empty or whitespace-only
    for (size_t i = 0; i < fields.size(); ++i)
    {
        if (fields[i].empty() || is_whitespace_only(fields[i]))
        {
            EmptyFieldException e;
            e.which_field = static_cast<unsigned int>(i);
            throw e;
        }
    }

    // Error check 3: Numeric fields must start with a digit
    const int numeric_idxs[] = {0, 3, 4, 5, 6, 7, 9, 10};
    for (int idx : numeric_idxs)
    {
        if (!begins_with_digit(fields[static_cast<size_t>(idx)]))
        {
            NonNumericDataException e;
            e.bad_field = fields[static_cast<size_t>(idx)];
            throw e;
        }
    }

    // Parse the fields
    auto to_int = [](std::string const &s)
    { return std::stoi(trim(s)); };

    int route = to_int(fields[0]);
    std::string source = trim(fields[1]);
    std::string dest = trim(fields[2]);
    int year = to_int(fields[3]);
    int month = to_int(fields[4]);
    int day = to_int(fields[5]);
    int hour = to_int(fields[6]);
    int minute = to_int(fields[7]);
    std::string vessel = trim(fields[8]);
    int expected = to_int(fields[9]);
    int actual = to_int(fields[10]);

    // Error check 4: Time must be valid (0-23 hours, 0-59 minutes)
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59)
    {
        InvalidTimeException e;
        e.bad_time = TimeOfDay{hour, minute};
        throw e;
    }

    // Build and return the Sailing object
    Sailing s;
    s.route_number = route;
    s.source_terminal = source;
    s.dest_terminal = dest;
    s.vessel_name = vessel;
    s.departure_date = Date{day, month, year};
    s.scheduled_departure_time = TimeOfDay{hour, minute};
    s.expected_duration = expected;
    s.actual_duration = actual;

    return s;
}

// Calculate stats for each route
std::vector<RouteStatistics> performance_by_route(std::vector<Sailing> const &sailings)
{
    std::vector<RouteStatistics> routes;

    for (auto const &s : sailings)
    {
        // Find if this route already exists
        bool found = false;
        for (auto &r : routes)
        {
            if (r.route_number == s.route_number)
            {
                r.total_sailings += 1;
                if (is_late(s.actual_duration, s.expected_duration))
                {
                    r.late_sailings += 1;
                }
                found = true;
                break;
            }
        }

        // If route not found, add new entry
        if (!found)
        {
            RouteStatistics new_route;
            new_route.route_number = s.route_number;
            new_route.total_sailings = 1;
            new_route.late_sailings = is_late(s.actual_duration, s.expected_duration) ? 1 : 0;
            routes.push_back(new_route);
        }
    }

    return routes;
}

// Find days with lowest late sailing ratio
std::vector<DayStatistics> best_days(std::vector<Sailing> const &sailings)
{
    auto days = compute_day_stats(sailings);
    if (days.empty())
        return {};

    // Find minimum late/total ratio using cross-multiplication
    int best_late = days[0].late_sailings;
    int best_total = days[0].total_sailings;

    for (auto const &d : days)
    {
        // Compare d.late/d.total vs best_late/best_total
        long long lhs = 1LL * d.late_sailings * best_total;
        long long rhs = 1LL * best_late * d.total_sailings;
        if (lhs < rhs)
        {
            best_late = d.late_sailings;
            best_total = d.total_sailings;
        }
    }

    // Collect all days matching the best ratio
    std::vector<DayStatistics> out;
    for (auto const &d : days)
    {
        if (1LL * d.late_sailings * best_total == 1LL * best_late * d.total_sailings)
        {
            out.push_back(d);
        }
    }
    return out;
}

// Find days with highest late sailing ratio
std::vector<DayStatistics> worst_days(std::vector<Sailing> const &sailings)
{
    auto days = compute_day_stats(sailings);
    if (days.empty())
        return {};

    // Find maximum late/total ratio
    int worst_late = days[0].late_sailings;
    int worst_total = days[0].total_sailings;

    for (auto const &d : days)
    {
        long long lhs = 1LL * d.late_sailings * worst_total;
        long long rhs = 1LL * worst_late * d.total_sailings;
        if (lhs > rhs)
        {
            worst_late = d.late_sailings;
            worst_total = d.total_sailings;
        }
    }

    // Collect all days matching the worst ratio
    std::vector<DayStatistics> out;
    for (auto const &d : days)
    {
        if (1LL * d.late_sailings * worst_total == 1LL * worst_late * d.total_sailings)
        {
            out.push_back(d);
        }
    }
    return out;
}

// Provided functions
std::vector<Sailing> read_sailings(std::string const &input_filename)
{
    std::vector<Sailing> all_sailings;
    std::ifstream input_file;
    input_file.open(input_filename);

    int valid_sailings{0};
    int total_lines{0};

    if (input_file.is_open())
    {
        std::string line;
        while (std::getline(input_file, line))
        {
            total_lines++;
            try
            {
                Sailing s{parse_sailing(line)};
                valid_sailings++;
                all_sailings.push_back(s);
            }
            catch (IncompleteLineException &e)
            {
                std::cout << "Line " << total_lines << " is invalid: ";
                std::cout << e.num_fields << " fields found." << std::endl;
            }
            catch (EmptyFieldException &e)
            {
                std::cout << "Line " << total_lines << " is invalid: ";
                std::cout << "Field " << e.which_field << " is empty." << std::endl;
            }
            catch (NonNumericDataException &e)
            {
                std::cout << "Line " << total_lines << " is invalid: ";
                std::cout << "\"" << e.bad_field << "\" is non-numeric." << std::endl;
            }
            catch (InvalidTimeException &e)
            {
                std::cout << "Line " << total_lines << " is invalid: ";
                std::cout << e.bad_time.hour << ":" << e.bad_time.minute << " is not a valid time." << std::endl;
            }
        }
        input_file.close();
    }
    else
    {
        throw std::runtime_error("Unable to open input file");
    }
    int invalid_sailings{total_lines - valid_sailings};
    std::cout << "Read " << valid_sailings << " records." << std::endl;
    std::cout << "Skipped " << invalid_sailings << " invalid records." << std::endl;
    return all_sailings;
}

void print_sailing(Sailing const &sailing)
{
    std::cout << "Route " << sailing.route_number;
    std::cout << " (" << sailing.source_terminal << " -> " << sailing.dest_terminal << "): ";
    std::cout << sailing.departure_date.year << "-";
    std::cout << std::setfill('0') << std::setw(2) << sailing.departure_date.month << "-";
    std::cout << std::setfill('0') << std::setw(2) << sailing.departure_date.day << " ";
    std::cout << std::setfill('0') << std::setw(2) << sailing.scheduled_departure_time.hour << ":";
    std::cout << std::setfill('0') << std::setw(2) << sailing.scheduled_departure_time.minute << " ";
    std::cout << "[Vessel: " << sailing.vessel_name << "] ";
    std::cout << sailing.actual_duration << " minutes (" << sailing.expected_duration << " expected)" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: ./assignment_2 action input_filename" << std::endl;
        std::cout << "       where action is either 'route_summary' or 'days'" << std::endl;
        return 1;
    }

    std::string action{argv[1]};
    std::string input_filename{argv[2]};

    auto all_sailings{read_sailings(input_filename)};

    if (action == "route_summary")
    {
        std::cout << "Performance by route:" << std::endl;
        auto statistics{performance_by_route(all_sailings)};
        for (auto stats : statistics)
        {
            std::cout << "Route " << stats.route_number << ": " << stats.total_sailings << " sailings (" << stats.late_sailings << " late)" << std::endl;
        }
    }
    else if (action == "days")
    {
        auto best{best_days(all_sailings)};
        auto worst{worst_days(all_sailings)};
        std::cout << "Best days:" << std::endl;
        for (auto stats : best)
        {
            std::cout << stats.date.year << "-" << stats.date.month << "-" << stats.date.day << ": ";
            std::cout << stats.total_sailings << " sailings (" << stats.late_sailings << " late)" << std::endl;
        }
        std::cout << "Worst days:" << std::endl;
        for (auto stats : worst)
        {
            std::cout << stats.date.year << "-" << stats.date.month << "-" << stats.date.day << ": ";
            std::cout << stats.total_sailings << " sailings (" << stats.late_sailings << " late)" << std::endl;
        }
    }
    else
    {
        std::cout << "Invalid action " << action << std::endl;
    }

    return 0;
}