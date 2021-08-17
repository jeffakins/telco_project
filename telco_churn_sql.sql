SELECT * 
FROM customers;

SELECT * 
FROM contract_types;
/* 
1	Month-to-month
2	One year
3	Two year 
*/

SELECT *
FROM internet_service_types;
/* 
1	DSL
2	Fiber optic
3	None 
*/

SELECT *
FROM payment_types;
/* 
1	Electronic check
2	Mailed check
3	Bank transfer (automatic)
4	Credit card (automatic) 
*/

SELECT * 
FROM customers
JOIN contract_types USING(contract_type_id)
JOIN internet_service_types USING(internet_service_type_id)
JOIN payment_types USING(payment_type_id);
