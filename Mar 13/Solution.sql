CREATE DATABASE mar_13;
USE mar_13;
# QUESTION 1
CREATE TABLE Product (
ID INT PRIMARY KEY);
CREATE TABLE Customer (
Customer_id INT,
Product_id INT, 
FOREIGN KEY (Product_id) REFERENCES  Product(ID)
);
INSERT INTO Product VALUES(5), (6);
INSERT INTO Customer VALUES (1,5),(2,6),(3,5),(3,6),(1,6);
SELECT * FROM Product;
SELECT * FROM Customer;
# SOLUTION 1
SELECT Customer_id
FROM Customer
GROUP BY Customer_id
HAVING COUNT(DISTINCT Product_id ) = (SELECT COUNT(*) FROM Product);
# QUESTIONS 2
CREATE TABLE Product_2 (
product_id INT ,
low_fats ENUM ('Y','N'),
recycleable ENUM ('Y','N')
);
INSERT INTO Product_2 VALUES  (0,'Y','N'),(1,'Y','Y'),(2,'N','Y'),(3,'Y','Y'),(4,'N','N'); 
SELECT * FROM Product_2;    
# SOLUTION 2
SELECT product_id FROM Product_2 
WHERE low_fats = 'Y'
AND recycleable = 'Y';

# QUESTION 3
CREATE TABLE Segments (
side1 INT, 
side2 INT, 
side3 INT);

INSERT INTO Segments VALUES (13, 15 , 30 ),(10 , 20 , 15 ); 

SELECT * FROM Segments;
# SOLUTION 3
SELECT side1 , side2, side3,
CASE WHEN side1 + side2 > side3
AND side1 + side3 > side2
AND side2 + side3 > side1
THEN 'YES'
ELSE 'NO'
END AS Trinagle
FROM Segments ;

# QUESTION 4

CREATE TABLE Friends (
friends_id INT PRIMARY KEY,
name VARCHAR (50),
activity VARCHAR (50));

CREATE TABLE Activities (
activity_id INT PRIMARY KEY,
name VARCHAR(50));

INSERT INTO Friends  VALUES 
(1,  'Jonathan D.', 'Eating'),(2 ,'Jade W.' ,'Singing'  ),
(3 , 'Victor J.','Singing'  ),(4 ,'Elvis Q.' ,'Eating' ),
(5    , 'Daniel A.','Eating'),(6,'Bob B.', 'Horse Riding') ;

INSERT INTO Activities VALUES (1, 'Eating'), (2, 'Singing'), (3, 'Horse Riding');


SELECT * FROM Activities ;
SELECT * FROM Friends ;

# SOLUTION 4
SELECT a.name AS activity
FROM Activities a
JOIN Friends f ON a.activity_id = f.activity_id
GROUP BY a.activity_id
HAVING COUNT(DISTINCT f.friends_id) != (
    SELECT MAX(participant_count) AS max_count
    FROM (
        SELECT COUNT(DISTINCT friends_id) AS participant_count
        FROM Friends
        GROUP BY activity_id
    ) AS max_counts
)
AND COUNT(DISTINCT f.friends_id) != (
    SELECT MIN(participant_count) AS min_count
    FROM (
        SELECT COUNT(DISTINCT friends_id) AS participant_count
        FROM Friends
        GROUP BY activity_id
    ) AS min_counts
);

# QUESTION 5
CREATE TABLE  Student_class (
student VARCHAR (50),
class VARCHAR (50)
);
INSERT INTO Student_class VALUES 
( 'A'       , 'Math'),       
( 'B'       , 'English'),    
( 'C'       , 'Math'     ),  
( 'D'       , 'Biology'    ),
( 'E'       , 'Math'       ),
( 'F'       , 'Computer'   ),
( 'G'       , 'Math'       ),
( 'H'       , 'Math'      ),
('I'       , 'Math'       );

# SOLUTION 5
SELECT class 
FROM Student_class
GROUP BY class
HAVING COUNT(student) >=5;

# QUESTION 6

CREATE TABLE Person (
person_id INT,
email VARCHAR(50)
);
INSERT INTO Person VALUES 
( 1  , 'a@b.com' ),
( 2  , 'c@d.com' ),
( 3  , 'a@b.com' );

# SOLUTION 6

SELECT email 
FROM Person 
GROUP BY email
HAVING COUNT(*) > 1; 

# QUESTION 7

CREATE TABLE Submissions (
sub_id INT,
parent_id INT);


INSERT INTO Submissions VALUES 

( 1       , Null       ),
( 2       , Null       ),
( 1       , Null       ),
( 12      , Null       ),
( 3       , 1          ),
( 5       , 2          ),
( 3       , 1          ),
( 4       , 1          ),
( 9       , 1          ),
( 10      , 2          ),
( 6       , 7  );

# SOLUTION 7

SELECT s.sub_id AS post_id,
COUNT(DISTINCT c.sub_id) AS number_of_comments 
FROM Submissions s LEFT JOIN Submissions c ON s.sub_id=c.parent_id 
WHERE s.parent_id IS NULL 
GROUP BY s.sub_id
ORDER BY s.sub_id;

