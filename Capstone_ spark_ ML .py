#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
Spark = (SparkSession.builder.appName("capstone")        .config("hive.metastore.uris","thrift://ip-10-1-2-24.ap-south-1.compute.internal:9083")        .enableHiveSupport().getOrCreate())


# In[2]:


Spark


# In[4]:


Spark.sql("select * from abinlab.employee").show()


# In[5]:


dept = Spark.sql("select * from abinlab.dept")


# In[9]:


Spark.sql("select s.emp_no, e.last_name, e.first_name, e.sex, s.salary from abinlab.employee as e inner join abinlab.Salaries as s on s.emp_no = e.emp_no order by s.emp_no").show()


# In[11]:


Spark.sql("select emp_no, last_name, first_name, hire_date from abinlab.employee where cast(substr( hire_date,7,4) as int) = '1986'").show() 


# In[22]:


Spark.sql("select distinct  department_Managers.dept_no, Departments.dept_name, Department_Managers.emp_no, employeesorg.last_name, employeesorg.first_name from mounica.Department_Managers inner join mounica.Departments on Department_Managers.dept_no= Departments.dept_no inner join mounica.employeesorg on Department_Managers.emp_no = employeesorg.emp_no order by Department_Managers.dept_no").show()


# In[24]:


Spark.sql("select distinct(m.dept_no),m.emp_no,d.dept_name,ee.first_name,ee.last_name FROM abinlab.dept_managers m inner join abinlab.dept d on m.dept_no = d.dept_no inner join abinlab.dept_emp e on m.dept_no = e.dept_no inner join abinlab.employee ee on m.emp_no = ee.emp_no").show()


# In[27]:


Spark.sql("select distinct(e.emp_no), e.last_name, e.first_name, dd.dept_name from abinlab.employee e INNER JOIN abinlab.dept_emp d on e.emp_no = d.emp_no INNER JOIN abinlab.dept dd on d.dept_no = dd.dept_no").show()


# In[28]:


employees = Spark.sql ( "select * from abinlab.employee")


# In[30]:


employees.show()


# In[33]:


Spark.sql("select first_name, last_name , sex from abinlab.employee where first_name ='Hercules' and last_name like 'B%'").show()


# In[44]:


Spark.sql("select distinct(e.emp_no), e.last_name, e.first_name, dd.dept_name from abinlab.employee e LEFT JOIN abinlab.dept_emp d on e.emp_no = d.emp_no LEFT JOIN abinlab.dept dd on d.dept_no = dd.dept_no where dd.dept_name like '%Sales%'").show()


# In[46]:


Spark.sql("select distinct(e.emp_no), e.last_name, e.first_name, dd.dept_name from abinlab.employee e LEFT JOIN abinlab.dept_emp d on e.emp_no = d.emp_no LEFT JOIN abinlab.dept dd on d.dept_no = dd.dept_no where dd.dept_name like '%Sales%' or dd.dept_name like '%development%'").show()


# In[47]:


Spark.sql("select last_name, count(last_name) freq from employee group by last_name order by freq desc ").show()


# In[23]:


Spark.sql("select salary_bins, count(emp_no) as frequency from abinlab.salary_dist group by salary_bins").show()


# In[17]:


avg = Spark.sql("select t.title, avg(s.salary) from employee e inner join titles t on e.emp_titles = t.title_id inner join salaries s on e.emp_no = s.emp_no group by t.title").toPandas()


# In[27]:


avg.columns = ['title', 'Average']


# In[30]:


avg


# In[46]:


sns.barplot(x = 'title', y ='Average', data = avg)
sns.set( {'figure.figsize':(7,14)})


# In[ ]:





# In[33]:


import seaborn as sns


# In[6]:


from pyspark.ml.feature import * 


# In[7]:


from pyspark.ml.classification import * 


# In[8]:


from pyspark.ml.evaluation import * 


# In[9]:


import numpy as np


# In[19]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


from pyspark.sql import functions as f 
from pyspark.sql import SparkSession, Window


# In[12]:


import pandas as pd 


# In[43]:


employee = Spark.sql("select * from employee")
titles = Spark.sql("select * from titles")
salaries = Spark.sql("select * from salaries")
dept= Spark.sql("select * from abinlab.dept")
dept_managers =Spark.sql("select * from dept_managers")
dept_employees = Spark.sql("select * from dept_employee")


# In[70]:


df = employee.join(dept_managers,on = 'emp_no', how = 'left')         .join(salaries, on ='emp_no', how = 'left')        .join (titles, titles.title_id == employee.emp_titles, how = 'left')
        


# In[71]:


df = df.join(dept, on ='dept_no', how = 'left')


# In[ ]:





# In[72]:


df.show(truncate = False)


# In[ ]:





# In[50]:


len(df.columns)


# In[ ]:


# Taking only the relevant columns 


# In[73]:


df=df['Sex','no_of_projects','last_performance_rating','dept_name','salary','title','left_org']


# In[80]:


df.select('dept_name').distinct().show()


# #converting categorical columns to labels 

# In[95]:


SI_dep_name = StringIndexer(inputCol='dept_name',outputCol='dept_name_labels', handleInvalid = 'skip' )


# In[96]:


df = SI_dep_name.fit(df).transform(df)


# In[65]:


type(df_final)


# In[88]:


SI_sex = StringIndexer(inputCol='Sex',outputCol='sex_index')


# In[89]:


df = SI_sex.fit(df).transform(df)


# In[97]:


df.show()


# In[98]:


SI_last_performance_rating = StringIndexer(inputCol='last_performance_rating',outputCol='last_performance_rating_freq')


# In[99]:


df = SI_last_performance_rating.fit(df).transform(df)


# In[100]:


df.show()


# ## Assembler vector

# In[101]:


assembler = VectorAssembler(inputCols = [ 'salary', 'no_of_projects','dept_name_labels','sex_index', 'last_performance_rating_freq'],outputCol = "features")


# In[102]:


output = assembler.transform(df)


# In[103]:


final_data = output.select('features','left_org')


# #encoding y variable 

# In[104]:


from pyspark.sql.functions import col,isnan, when, count
final_data = final_data.withColumn('label',when(final_data.left_org == '1',1).otherwise(0))


# ### Splitting Data 

# In[105]:


train_df,test_df = final_data.randomSplit( [0.7, 0.3], seed = 50 )


# ### Logistic Regression 

# In[106]:


logrg = LogisticRegression(featuresCol= 'features',labelCol='label',maxIter=5)


# In[107]:


lr = logrg.fit(train_df)


# In[108]:


y_pred_test = lr.transform(test_df)


# In[109]:


y_pred_train = lr.transform(train_df)


# In[110]:


y_pred_train.select(['label',
 'rawPrediction',
 'probability',
 'prediction']).toPandas().head(10)


# ### Precision

# In[111]:


y_pred_test.filter(y_pred_test.label == y_pred_test.prediction).count() / float(y_pred_test.count())


# In[ ]:




