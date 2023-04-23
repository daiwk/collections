select
    channel_id,
    reason,
    sum(p) as p,
    sum(ic) as ic,
    percentile_approx(combined_score,array(0. ,0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95,1)) as combined_score,
    percentile_approx(score1,array(0. ,0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95,1)) as score1,
    percentile_approx(score2,array(0. ,0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95,1)) as score2
    
from 
(select
     channel_id,
     get_json_object(string_list_params,'$.r_reasons.list') as reason,
     get_json_object(double_params,'$.combined_score') as combined_score,
     get_json_object(double_params,'$.score1') as score1,
     get_json_object(double_params,'$.score2') as score2,
    case when get_json_object(int_params, '$.p_i') is not null then 1 else 0 end as p,
    case
            when get_json_object(int_params, '$.ic_i') is not null then 1
            else 0
        end as ic
     
from table_xx
where date between '20230304' and '20230322'
and channel_id in (123)
and get_json_object(string_list_params,'$.r_reasons.list') rlike 'recall1|recall2'
and  size(split(get_json_object(string_list_params,'$.r_reasons.list'),',')) = 2 
and get_json_object(double_params,'$.combined_score') is not null 
and ab_versions rlike '333'
) tmp 
group by channel_id,reason 
