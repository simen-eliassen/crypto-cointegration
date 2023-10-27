drop table tblModels;
CREATE TABLE tblModels (
    model_id INT PRIMARY KEY,
    model_name VARCHAR(50),
    model_type VARCHAR(50),
    model_coeff varchar(1000),
    model_date DATETIME,
    training_from_date DATETIME,
    training_to_date DATETIME
);