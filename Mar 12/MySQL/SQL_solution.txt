CREATE DATABASE mar_12;
USE mar_12;
CREATE TABLE Employee (
id INT PRIMARY KEY,
salary INT);

CREATE TABLE Calls (
from_id INT, 
to_id INT,
duration INT);

CREATE TABLE Student (
name VARCHAR(50),
continent VARCHAR(50));

INSERT INTO Employee VALUES 
(1, 100),
(2, 200),
(3, 300);

INSERT INTO Calls VALUES 
    (1, 2, 59),
    (2, 1, 11),
    (1, 3, 20),
    (3, 4, 100),
    (3, 4, 200),
    (3, 4, 200),
    (4, 3, 499);
    
INSERT INTO Student VALUES 
('Jack','America'),   
('Pascal','Europe'),    
('Xi','Asia'),      
('Jane','America');   

# solution 1
SELECT (
SELECT DISTINCT Salary
FROM Employee 
ORDER BY Salary DESC
LIMIT 1 OFFSET 1
) AS second_highest_salary;

# solution 2
SELECT CASE WHEN from_id < to_id THEN from_id ELSE to_id END AS person1,
	   CASE WHEN from_id < to_id THEN to_id ELSE from_id END AS person2,
	   COUNT(*) AS call_count,
	   SUM(duration) AS total_duration
FROM Calls
GROUP BY person1, person2
ORDER BY person1, person2;

# solution 3
WITH RankedStudents AS (
    SELECT
        name,
        continent,
        ROW_NUMBER() OVER (PARTITION BY continent ORDER BY name) AS row_num
    FROM student
)

SELECT
    MAX(CASE WHEN continent = 'America' THEN name END) AS America,
    MAX(CASE WHEN continent = 'Asia' THEN name END) AS Asia,
    MAX(CASE WHEN continent = 'Europe' THEN name END) AS Europe
FROM RankedStudents
GROUP BY row_num
ORDER BY row_num;