with sub_query as(select q.the_date,q.weather_type,q.previous_weather_type,q.difference,sum(difference) over(order by q.the_date
																							rows between unbounded preceding and current row) as comulative
from(
	select t.the_date,t.weather_type,t.previous_weather_type,case when t.weather_type=t.previous_weather_type then 0
																  when t.weather_type!=t.previous_weather_type then 1
																  else 0 end as difference
	from(
		select the_date,weather_type,lag(weather_type) over() as previous_weather_type
		from weather
		) as t
		) as q
)
select weather_type,comulative,count(*) as number_of_days
from sub_query
group by 2,1
order by 3 desc

