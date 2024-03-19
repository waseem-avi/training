USE mar_14;
# SOLUTION 1
SELECT FIRST_NAME, LAST_NAME , HIRE_DATE 
FROM employees 
WHERE HIRE_DATE > (SELECT HIRE_DATE 
FROM employees WHERE LAST_NAME = 'Jones');

# SOLUTION 2

SELECT d.DEPARTMENT_NAME , e.EMPLOYEE_NUMBER 
FROM departments AS d 
JOIN 
(SELECT DEPARTMENT_ID , COUNT(*) AS EMPLOYEE_NUMBER
FROM employees
GROUP BY DEPARTMENT_ID) AS e
ON d.DEPARTMENT_ID = e.DEPARTMENT_ID;

# SOLUTION 3
CREATE TABLE job_history (
    employee_id INT,
    start_date DATE,
    end_date DATE,
    job_id VARCHAR(10),
    department_id INT
);

-- Insert sample data into the job_history table
INSERT INTO job_history (employee_id, start_date, end_date, job_id, department_id)
VALUES
(102, '1993-01-13', '1998-07-24', 'IT_PROG', 60),
(101, '1989-09-21', '1993-10-27', 'AC_ACCOUNT', 110),
(101, '1993-10-28', '1997-03-15', 'AC_MGR', 110),
(201, '1996-02-17', '1999-12-19', 'MK_REP', 20),
(114, '1998-03-24', '1999-12-31', 'ST_CLERK', 50),
(122, '1999-01-01', '1999-12-31', 'ST_CLERK', 50),
(200, '1987-09-17', '1993-06-17', 'AD_ASST', 90),
(176, '1998-03-24', '1998-12-31', 'SA_REP', 80),
(176, '1999-01-01', '1999-12-31', 'SA_MAN', 80),
(200, '1994-07-01', '1998-12-31', 'AC_ACCOUNT', 90);

-- Create the jobs table
CREATE TABLE jobs (
    JOB_ID VARCHAR(10),
    JOB_TITLE VARCHAR(50),
    MIN_SALARY INT,
    MAX_SALARY INT
);

-- Insert sample data into the jobs table
INSERT INTO jobs (JOB_ID, JOB_TITLE, MIN_SALARY, MAX_SALARY)
VALUES
('AD_PRES', 'President', 20000, 40000),
('AD_VP', 'Administration Vice President', 15000, 30000),
('AD_ASST', 'Administration Assistant', 3000, 6000),
('FI_MGR', 'Finance Manager', 8200, 16000),
('FI_ACCOUNT', 'Accountant', 4200, 9000),
('AC_MGR', 'Accounting Manager', 8200, 16000),
('AC_ACCOUNT', 'Public Accountant', 4200, 9000),
('SA_MAN', 'Sales Manager', 10000, 20000),
('SA_REP', 'Sales Representative', 6000, 12000),
('PU_MAN', 'Purchasing Manager', 8000, 15000),
('PU_CLERK', 'Purchasing Clerk', 2500, 5500),
('ST_MAN', 'Stock Manager', 5500, 8500),
('ST_CLERK', 'Stock Clerk', 2000, 5000),
('SH_CLERK', 'Shipping Clerk', 2500, 5500),
('IT_PROG', 'Programmer', 4000, 10000),
('MK_MAN', 'Marketing Manager', 9000, 15000),
('MK_REP', 'Marketing Representative', 4000, 9000),
('HR_REP', 'Human Resources Representative', 4000, 9000),
('PR_REP', 'Public Relations Representative', 4500, 10500);

# SOLUTION 3
-- 3.Write a MySQL query to find the employee ID, job title, 
-- number of days between ending date and starting date for all 
-- jobs in department 90 from job history.

SELECT h.employee_id , h.days_num , j.JOB_TITLE 
FROM jobs AS j
JOIN ( SELECT employee_id , end_date - start_date AS days_num , job_id 
FROM job_history 
WHERE department_id = 90) AS h
ON h.job_id = j.job_id; 

# SOLUTION 4

SELECT d.DEPARTMENT_ID , d.DEPARTMENT_NAME , d.MANAGER_ID , e.FIRST_NAME
FROM departments as d
JOIN employees as e
ON d.MANAGER_ID = e.EMPLOYEE_ID;

# SOLUTION 5

SELECT d.DEPARTMENT_ID , d.DEPARTMENT_NAME, e.FIRST_NAME , l.CITY 
FROM departments as d
JOIN locations as l
ON d.LOCATION_ID = l.LOCATION_ID
JOIN employees as e
ON d.MANAGER_ID = e.EMPLOYEE_ID; 

# SOLUTION 6
SELECT j.JOB_TITLE, ROUND(AVG(e.SALARY)) AS AVG_SALARY 
FROM jobs AS j 
JOIN employees AS e ON e.JOB_ID = j.JOB_ID 
GROUP BY j.JOB_TITLE;