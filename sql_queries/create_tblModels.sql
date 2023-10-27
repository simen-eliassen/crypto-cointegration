--drop table if exists tblModels;
create table tblModels(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_coeff varchar(1000) NOT NULL,
    model_date DATETIME NOT NULL,
    training_from_date DATETIME NOT NULL,
    training_to_date DATETIME NOT NULL
);