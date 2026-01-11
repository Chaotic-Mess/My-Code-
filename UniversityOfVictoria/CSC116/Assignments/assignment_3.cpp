/*
 * Program: C++ Assignment 3 - Student Database
 * Author: Zac Matthias
 * Date: December 16th 2025
 * Description:
 *   This program contains multiple exercises demonstrating C++ object-oriented
 *   programming concepts, including classes, STL containers, and exception handling:
 *     1. Exercise 1 : Implementation of a student database supporting enrollment,
 *                     grading, transcript generation, and average calculations.
 *
 * Libraries used:
 *   - <iostream> for input/output (cin, cout)
 *   - <string> for string manipulation
 *   - <set> for storing unique collections of items
 *   - <map> for key-value pair storage
 *
 *   PS E:\CSC116> g++ assignment3.cpp -o assignment3
 *   PS E:\CSC116> ./assignment3
 */

/*  Definitions for the StudentDB class.

   You may modify this file, but ONLY in the places indicated. In particular, you must
   place the implementation of each function.

   In the specification comments below, the following terms are used.
    - A "student ID" is a string identifier for a student. An example might be
      "V00123456", but any string could reasonably be used.
    - A "course ID" is a string identifier for a course. We might use something
      like "CSC 116", but again, any string is feasible.
    - A "term code" is a string representing a particular academic term. As
      with student and course IDs, a term code can be any string, but often
      we might choose a string representing a date in "YYYYMM" notation,
      like "202209".

   In each of the function specifications below, if there is a list of
   error cases provided, your code MUST check for the error cases in the
   order written. If a particular call to the function would result in
   more than one error condition being met, the exception thrown must be
   the error condition listed FIRST in the specification comment.
*/

#include <string>
#include <set>
#include <map>
#include <iostream>

/* Definitions of exception objects */
/* Do not modify these definitions in any way */

/* DBDuplicateError: Thrown when attempts are made to insert an already-existing
                     record into the database. */
struct DBDuplicateError
{
    // This structure has no members, since in the contexts where it is thrown,
    // the cause of the error will be unambiguous.
};

/* StudentNotFoundError: Thrown in cases where a function is called with a
                         student ID that has not yet been added to the
                         database. */
struct StudentNotFoundError
{
    std::string student_id{};
};

/* EnrollmentNotFoundError: Thrown when a student enrollment record is expected
                            to exist but does not exist. */
struct EnrollmentNotFoundError
{
    std::string student_id{};
    std::string course_id{};
    std::string term{};
};

/* InvalidGradeError: Thrown when an invalid grade value (greater than 100)
                      is provided to a database function. */
struct InvalidGradeError
{
    unsigned int bad_grade{}; // The grade value that produced the error
};

/* MissingGradeError: Thrown when a non-existant grade is requested. */
struct MissingGradeError
{
    // This has no members.
};

/* EmptyAverageError: Thrown when an average is requested but no source
                      grades exist. */
struct EmptyAverageError
{
    // This has no members.
};

/* Definition of the StudentDB class */
/* You are only permitted to modify the private section of this class definition. */
class StudentDB
{
public:
    /* Do not modify any of the declarations and code in this section in any way. */
    /* (Impelment these function below) */ 
    // THESE ARE PROTOTYPES
    StudentDB();
    void add_student(std::string const &student_id);
    std::set<std::string> all_students();
    void enroll(std::string const &student_id, std::string const &course_id, std::string const &term);
    std::set<std::pair<std::string, std::string>> get_student_enrollment_records(std::string const &student_id);
    std::set<std::string> courses_taken_by_student(std::string const &student_id);
    void assign_grade(std::string const &student_id, std::string const &course_id, std::string const &term, unsigned int grade);
    unsigned int get_grade(std::string const &student_id, std::string const &course_id, std::string const &term);
    std::map<std::string, unsigned int> student_transcript_by_course(std::string const &student_id);
    double compute_student_average(std::string const &student_id);
    std::set<std::string> enrolled_students(std::string const &course_id, std::string const &term);
    std::map<std::string, unsigned int> course_grades(std::string const &course_id, std::string const &term);
    double compute_course_average(std::string const &course_id, std::string const &term);

private:
    std::set<std::string> students_{};                                                            // Students known to the DB
    std::map<std::string, std::set<std::pair<std::string, std::string>>> enroll_by_student_{};    // Enrollment records by student: student_id -> set of (course_id, term)
    std::map<std::pair<std::string, std::string>, std::set<std::string>> enroll_by_offering_{};   // Enrollment records by offering: (course_id, term) -> set of student_ids
    std::map<std::tuple<std::string, std::string, std::string>, unsigned int> grades_{};          // Grades by enrollment: (student_id, course_id, term) -> grade


    bool has_student_(std::string const &student_id) const
    {
        return students_.find(student_id) != students_.end();
    }

    bool has_enrollment_(std::string const &student_id,
                         std::string const &course_id,
                         std::string const &term) const
    {
        auto it = enroll_by_student_.find(student_id);
        if (it == enroll_by_student_.end()) return false;
        return it->second.find({course_id, term}) != it->second.end();
    }
};

StudentDB::StudentDB()
{
    // Start empty just in case
    students_.clear();
    enroll_by_student_.clear();
    enroll_by_offering_.clear();
    grades_.clear();
}

void StudentDB::add_student(std::string const &student_id)
{
    if (students_.find(student_id) != students_.end())
        throw DBDuplicateError{};

    students_.insert(student_id);
    enroll_by_student_.emplace(student_id, std::set<std::pair<std::string, std::string>>{}); 
    // Ensure the student has an enrollment bucket (optional but convenient)
}

std::set<std::string> StudentDB::all_students()
{
    return students_;
}

void StudentDB::enroll(std::string const &student_id, std::string const &course_id, std::string const &term)
{
    // Error checks MUST be in this order per instructions:
    if (has_enrollment_(student_id, course_id, term)) // 1) duplicate enrollment
        throw DBDuplicateError{};

    if (!has_student_(student_id)) // 2) invalid student
        throw StudentNotFoundError{student_id};

    enroll_by_student_[student_id].insert({course_id, term});
    enroll_by_offering_[{course_id, term}].insert(student_id);
}

std::set<std::pair<std::string, std::string>>
StudentDB::get_student_enrollment_records(std::string const &student_id)
{
    if (!has_student_(student_id))
        throw StudentNotFoundError{student_id};

    auto it = enroll_by_student_.find(student_id);
    if (it == enroll_by_student_.end())
        return {};

    return it->second;
}

std::set<std::string> StudentDB::courses_taken_by_student(std::string const &student_id)
{
    if (!has_student_(student_id))
        throw StudentNotFoundError{student_id};

    std::set<std::string> result{};
    auto records = get_student_enrollment_records(student_id);
    for (auto const &p : records)
    {
        result.insert(p.first); // course_id
    }
    return result;
}

void StudentDB::assign_grade(std::string const &student_id,
                            std::string const &course_id,
                            std::string const &term,
                            unsigned int grade)
{
    // Error checks MUST be in this order per spec:
    // 1) invalid student
    if (!has_student_(student_id))
        throw StudentNotFoundError{student_id};

    // 2) not enrolled
    if (!has_enrollment_(student_id, course_id, term))
        throw EnrollmentNotFoundError{student_id, course_id, term};

    // 3) invalid grade (> 100)
    if (grade > 100)
        throw InvalidGradeError{grade};

    grades_[std::make_tuple(student_id, course_id, term)] = grade;
}

unsigned int StudentDB::get_grade(std::string const &student_id,
                                 std::string const &course_id,
                                 std::string const &term)
{
    // Error checks MUST be in this order per spec:
    // 1) invalid student
    if (!has_student_(student_id))
        throw StudentNotFoundError{student_id};

    // 2) not enrolled
    if (!has_enrollment_(student_id, course_id, term))
        throw EnrollmentNotFoundError{student_id, course_id, term};

    // 3) missing grade
    auto key = std::make_tuple(student_id, course_id, term);
    auto it = grades_.find(key);
    if (it == grades_.end())
        throw MissingGradeError{};

    return it->second;
}

std::map<std::string, unsigned int>
StudentDB::student_transcript_by_course(std::string const &student_id)
{
    if (!has_student_(student_id))
        throw StudentNotFoundError{student_id};

    std::map<std::string, unsigned int> best_by_course{};

    // Only include courses with at least one assigned grade; keep highest per course.
    auto records = get_student_enrollment_records(student_id);
    for (auto const &[course, term] : records)
    {
        auto key = std::make_tuple(student_id, course, term);
        auto itg = grades_.find(key);
        if (itg == grades_.end()) continue;

        auto itbest = best_by_course.find(course);
        if (itbest == best_by_course.end() || itg->second > itbest->second)
            best_by_course[course] = itg->second;
    }

    return best_by_course;
}

double StudentDB::compute_student_average(std::string const &student_id)
{
    if (!has_student_(student_id))
        throw StudentNotFoundError{student_id};

    auto transcript = student_transcript_by_course(student_id);
    if (transcript.empty())
        throw EmptyAverageError{};

    double sum = 0.0;
    for (auto const &kv : transcript)
        sum += static_cast<double>(kv.second);

    return sum / static_cast<double>(transcript.size());
}

std::set<std::string> StudentDB::enrolled_students(std::string const &course_id, std::string const &term)
{
    auto it = enroll_by_offering_.find({course_id, term});
    if (it == enroll_by_offering_.end())
        return {};

    return it->second;
}

std::map<std::string, unsigned int> StudentDB::course_grades(std::string const &course_id, std::string const &term)
{
    std::map<std::string, unsigned int> result{};

    auto enrolled = enrolled_students(course_id, term);
    if (enrolled.empty())
        return result; // "no entries" => empty map

    for (auto const &student_id : enrolled)
    {
        auto key = std::make_tuple(student_id, course_id, term);
        auto itg = grades_.find(key);
        if (itg != grades_.end())
            result[student_id] = itg->second; // only those with assigned grades
    }

    return result;
}

double StudentDB::compute_course_average(std::string const &course_id, std::string const &term)
{
    auto gmap = course_grades(course_id, term);
    if (gmap.empty())
        throw EmptyAverageError{};

    double sum = 0.0;
    for (auto const &kv : gmap)
        sum += static_cast<double>(kv.second);

    return sum / static_cast<double>(gmap.size());
}


int main()
{
    StudentDB db{};

    std::cout << "Test 1: Adding some students" << std::endl;
    db.add_student("V00123458");
    db.add_student("V00123457");
    db.add_student("V00123456");

    std::cout << "  Test 1a: Attempting to create a duplicate student" << std::endl;
    try
    {
        db.add_student("V00123456");
        std::cout << "    Did not catch DBDuplicateError: Behaviour is incorrect." << std::endl;
    }
    catch (DBDuplicateError &e)
    {
        std::cout << "    Caught DBDuplicateError: Behaviour is correct." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 2: Retrieving student list." << std::endl;
    {
        std::cout << "  Students are: ";

        std::set<std::string> all_students{db.all_students()};
        for (auto s : all_students)
        {
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 3: Enrolling students in courses." << std::endl;
    db.enroll("V00123456", "CSC 116", "202209");
    db.enroll("V00123456", "MECH 200", "202209");
    db.enroll("V00123456", "MATH 204", "202209");

    db.enroll("V00123457", "CSC 116", "202209");
    db.enroll("V00123457", "CSC 116", "202201");
    db.enroll("V00123457", "MATH 204", "202209");

    db.enroll("V00123458", "CSC 116", "202201");

    std::cout << "  Test 3a: Attempting to enroll an invalid student." << std::endl;
    try
    {
        db.enroll("V00999999", "CSC XYZ", "222222"); // The only invalid part of this call is the student ID (which is not in the database yet)
        std::cout << "    Did not catch StudentNotFoundError: Behaviour is incorrect." << std::endl;
    }
    catch (StudentNotFoundError &e)
    {
        std::cout << "    Caught StudentNotFoundError (" << e.student_id << "): Behaviour is correct." << std::endl;
    }

    std::cout << "  Test 3b: Attempting a duplicate enrollment." << std::endl;
    try
    {
        db.enroll("V00123457", "MATH 204", "202209");
        std::cout << "    Did not catch DBDuplicateError: Behaviour is incorrect." << std::endl;
    }
    catch (DBDuplicateError &e)
    {
        std::cout << "    Caught DBDuplicateError: Behaviour is correct." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 4: Retrieving enrollment records." << std::endl;
    {
        std::set<std::pair<std::string, std::string>> enrollment_set{};
        enrollment_set = db.get_student_enrollment_records("V00123456");
        std::cout << "  For V00123456:" << std::endl;
        for (auto [course, term] : enrollment_set)
        {
            std::cout << "    Course: " << course << "  Term: " << term << std::endl;
        }

        enrollment_set = db.get_student_enrollment_records("V00123457");
        std::cout << "  For V00123457:" << std::endl;
        for (auto [course, term] : enrollment_set)
        {
            std::cout << "    Course: " << course << "  Term: " << term << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Test 5: Retrieving courses taken by students." << std::endl;
    {
        std::set<std::string> courses_taken{};
        courses_taken = db.courses_taken_by_student("V00123456");
        std::cout << "  For V00123456: ";
        for (auto course : courses_taken)
        {
            std::cout << course << " ";
        }
        std::cout << std::endl;

        courses_taken = db.courses_taken_by_student("V00123457");
        std::cout << "  For V00123457: ";
        for (auto course : courses_taken)
        {
            std::cout << course << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 6: Adding more students and enrollments" << std::endl;
    db.add_student("V00123459");
    db.add_student("V00123460");

    db.enroll("V00123459", "CSC 116", "202209");
    db.enroll("V00123460", "MATH 204", "202209");
    db.enroll("V00123460", "MATH 342", "202209");

    std::cout << std::endl;
    std::cout << "Test 7: Assigning some grades" << std::endl;

    db.assign_grade("V00123456", "CSC 116", "202209", 67);
    db.assign_grade("V00123456", "MECH 200", "202209", 85);
    db.assign_grade("V00123456", "MATH 204", "202209", 92);

    db.assign_grade("V00123457", "CSC 116", "202201", 93);
    db.assign_grade("V00123457", "CSC 116", "202209", 87);

    db.assign_grade("V00123459", "CSC 116", "202209", 78);
    db.assign_grade("V00123460", "MATH 204", "202209", 90);

    std::cout << "  Test 7a: Attempting to grade an invalid student." << std::endl;
    try
    {
        db.assign_grade("V00999999", "CSC 116", "202209", 17);
        std::cout << "    Did not catch StudentNotFoundError: Behaviour is incorrect." << std::endl;
    }
    catch (StudentNotFoundError &e)
    {
        std::cout << "    Caught StudentNotFoundError (" << e.student_id << "): Behaviour is correct." << std::endl;
    }
    std::cout << "  Test 7b: Attempting to grade an unenrolled student." << std::endl;
    try
    {
        db.assign_grade("V00123458", "CSC 116", "202209", 17);
        std::cout << "    Did not catch EnrollmentNotFoundError: Behaviour is incorrect." << std::endl;
    }
    catch (EnrollmentNotFoundError &e)
    {
        std::cout << "    Caught EnrollmentNotFoundError (";
        std::cout << e.student_id << ", " << e.course_id << ", " << e.term;
        std::cout << "): Behaviour is correct." << std::endl;
    }
    std::cout << "  Test 7c: Attempting to assign an invalid grade." << std::endl;
    try
    {
        db.assign_grade("V00123458", "CSC 116", "202201", 187);
        std::cout << "    Did not catch InvalidGradeError: Behaviour is incorrect." << std::endl;
    }
    catch (InvalidGradeError &e)
    {
        std::cout << "    Caught InvalidGradeError (" << e.bad_grade << "): Behaviour is correct." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 8: Retrieving student grades." << std::endl;
    std::cout << "  Grade for V00123456 in CSC 116 in 202209: " << db.get_grade("V00123456", "CSC 116", "202209") << std::endl;
    std::cout << "  Grade for V00123457 in CSC 116 in 202201: " << db.get_grade("V00123457", "CSC 116", "202201") << std::endl;
    std::cout << "  Grade for V00123457 in CSC 116 in 202209: " << db.get_grade("V00123457", "CSC 116", "202209") << std::endl;

    std::cout << "  Test 8a: Attempting to retrieve a grade for an unenrolled student." << std::endl;
    try
    {
        ;
        auto gr{db.get_grade("V00123458", "CSC 116", "202209")};
        std::cout << "    Grade for V00123458 for CSC 116 in 202209: " << gr << std::endl;
        std::cout << "    Did not catch EnrollmentNotFoundError: Behaviour is incorrect." << std::endl;
    }
    catch (EnrollmentNotFoundError &e)
    {
        std::cout << "   Caught EnrollmentNotFoundError (";
        std::cout << e.student_id << ", " << e.course_id << ", " << e.term;
        std::cout << "): Behaviour is correct." << std::endl;
    }
    std::cout << "  Test 8b: Attempting to retrieve an unassigned grade." << std::endl;
    try
    {
        auto gr{db.get_grade("V00123458", "CSC 116", "202201")};
        std::cout << "    Grade for V00123458 for CSC 116 in 202201: " << gr << std::endl;
        std::cout << "    Did not catch MissingGradeError: Behaviour is incorrect." << std::endl;
    }
    catch (MissingGradeError &e)
    {
        std::cout << "    Caught MissingGradeError: Behaviour is correct." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 9: Retrieving transcripts." << std::endl;
    {
        std::map<std::string, unsigned int> transcript_map{};
        transcript_map = db.student_transcript_by_course("V00123456");
        std::cout << "  For V00123456:" << std::endl;
        for (auto [course, grade] : transcript_map)
        {
            std::cout << "    Course: " << course << "  Grade: " << grade << std::endl;
        }

        transcript_map = db.student_transcript_by_course("V00123457");
        std::cout << "  For V00123457:" << std::endl;
        for (auto [course, grade] : transcript_map)
        {
            std::cout << "    Course: " << course << "  Grade: " << grade << std::endl;
        }

        transcript_map = db.student_transcript_by_course("V00123458");
        if (transcript_map.size() == 0)
        {
            std::cout << "  For V00123458: Transcript is empty: Behaviour is correct." << std::endl;
        }
        else
        {
            std::cout << "  For V00123458:" << std::endl;
            for (auto [course, grade] : transcript_map)
            {
                std::cout << "    Course: " << course << "  Grade: " << grade << std::endl;
            }
            std::cout << "  This is incorrect (the transcript should be empty)." << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Test 10: Computing student averages." << std::endl;
    std::cout << "  Average for V00123456: " << db.compute_student_average("V00123456") << std::endl;
    std::cout << "  Average for V00123457: " << db.compute_student_average("V00123457") << std::endl;

    std::cout << "  Test 10a: Attempting to compute average for a student with no grades." << std::endl;
    try
    {
        auto av{db.compute_student_average("V00123458")};
        std::cout << "    Average for V00123458: " << av << std::endl;
        std::cout << "    Did not catch EmptyAverageError: Behavior is incorrect" << std::endl;
    }
    catch (EmptyAverageError &e)
    {
        std::cout << "    Caught EmptyAverageError: Behaviour is correct." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 11: Retrieving enrollment by course." << std::endl;
    {
        std::set<std::string> enrollment_set{};
        enrollment_set = db.enrolled_students("CSC 116", "202201");
        std::cout << "  For CSC 116 in 202201: ";
        for (auto student_id : enrollment_set)
        {
            std::cout << student_id << " ";
        }
        std::cout << std::endl;

        enrollment_set = db.enrolled_students("CSC 116", "202209");
        std::cout << "  For CSC 116 in 202209: ";
        for (auto student_id : enrollment_set)
        {
            std::cout << student_id << " ";
        }
        std::cout << std::endl;

        enrollment_set = db.enrolled_students("MATH 204", "202209");
        std::cout << "  For MATH 204 in 202209: ";
        for (auto student_id : enrollment_set)
        {
            std::cout << student_id << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Test 12: Retrieving grades by course." << std::endl;
    {
        std::map<std::string, unsigned int> grade_map{};
        grade_map = db.course_grades("CSC 116", "202201");
        std::cout << "  For CSC 116 in 202201:" << std::endl;
        for (auto [student_id, grade] : grade_map)
        {
            std::cout << "    Student: " << student_id << "  Grade: " << grade << std::endl;
        }

        grade_map = db.course_grades("CSC 116", "202209");
        std::cout << "  For CSC 116 in 202209:" << std::endl;
        for (auto [student_id, grade] : grade_map)
        {
            std::cout << "    Student: " << student_id << "  Grade: " << grade << std::endl;
        }

        grade_map = db.course_grades("MATH 204", "202209");
        std::cout << "  For MATH 204 in 202209:" << std::endl;
        for (auto [student_id, grade] : grade_map)
        {
            std::cout << "    Student: " << student_id << "  Grade: " << grade << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Test 13: Computing course averages." << std::endl;
    std::cout << "  Average for CSC 116 in 202201: " << db.compute_course_average("CSC 116", "202201") << std::endl;
    std::cout << "  Average for CSC 116 in 202209: " << db.compute_course_average("CSC 116", "202209") << std::endl;

    std::cout << "  Test 13a: Attempting to compute average for a course with no grades." << std::endl;
    try
    {
        auto av{db.compute_course_average("MATH 342", "202209")};
        std::cout << "    Average for MATH 342 in 202209: " << av << std::endl;
        std::cout << "    Did not catch EmptyAverageError: Behavior is incorrect" << std::endl;
    }
    catch (EmptyAverageError &e)
    {
        std::cout << "    Caught EmptyAverageError: Behaviour is correct." << std::endl;
    }

    return 0;
}