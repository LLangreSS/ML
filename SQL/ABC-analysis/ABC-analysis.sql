select album_name,sales,percentage,sum_,case when sum_ between 0 and 80 then 'A'
											 when sum_ between 80 and 95 then 'B' else 'C' end as category
											 
from(
	select album_name,sales,percentage,sum(percentage) over(order by sales desc
															rows between unbounded preceding and current row) as sum_
	from album_percentages
	)
