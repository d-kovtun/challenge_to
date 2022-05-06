#Left Join um keine Fälle mit impressions aber ohne clicks zu verlieren.
#Wir verlieren doch die Entries mit clicks und ohne impressions, aber es is unmöglich (oder es gibt ein Bug in Reporting).
#Ich habe gecheckt, dass CTR > 1 ist möglich.

# CTR is the number of clicks that your ad receives divided by the number of times your ad is shown: clicks ÷ impressions = CTR. 


### Simple

SELECT t1.teaser_id, IFNULL(click_cnt/impr_cnt, 0) AS CTR FROM
(
	SELECT teaser_id, count(*) impr_cnt FROM test.aufgabe
	WHERE YEAR(`timestamp`) = 2020 and event_name = "impression"
	GROUP BY teaser_id
) t1
LEFT JOIN 
(
	SELECT teaser_id, count(*) click_cnt FROM test.aufgabe
	WHERE YEAR(`timestamp`) = 2020 and event_name = "click"
	GROUP BY teaser_id
    ) t2
USING (teaser_id)
;

### For all years, more resource costly, since other years are also calculated, even if unused 

SELECT t1.teaser_id, t1.year_1 as YEAR, IFNULL(t2.click_cnt/t1.impr_cnt, 0) AS CTR FROM
(
	SELECT teaser_id, YEAR(`timestamp`) as year_1, count(*) impr_cnt FROM test.aufgabe
	WHERE  event_name = "impression"
	GROUP BY teaser_id, year_1
) t1
LEFT JOIN  
(
	SELECT teaser_id, YEAR(`timestamp`) as year_1, count(*) click_cnt FROM test.aufgabe
	WHERE  event_name = "click"
	GROUP BY teaser_id, year_1
    ) t2
ON t1.teaser_id = t2.teaser_id 
AND t1.year_1 = t2.year_1
HAVING YEAR = 2020; 

### For stored procedures calculation (be aware @ variables are shared in the session in MySQL):
SET @year_y = 2020;

SELECT t1.teaser_id, IFNULL(click_cnt/impr_cnt, 0) AS CTR FROM
(
	SELECT teaser_id, count(*) impr_cnt FROM test.aufgabe
	WHERE YEAR(`timestamp`) = @year_y  and event_name = "impression"
	GROUP BY teaser_id
) t1
LEFT JOIN 
(
	SELECT teaser_id, count(*) click_cnt FROM test.aufgabe
	WHERE YEAR(`timestamp`) = @year_y  and event_name = "click"
	GROUP BY teaser_id
    ) t2
ON t1.teaser_id = t2.teaser_id
;