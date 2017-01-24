from __future__ import print_function

from pyspark import SparkContext


from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import math
import itertools
from operator import add

sc = SparkContext(appName="CourseRecommendor")
sc.setCheckpointDir("checkpoint/")
course_grade_data = sc.textFile("2006-2016-course-prof-ratings.csv")
course_data_header = course_grade_data.take(1)[0]
course_grades = course_grade_data.filter(lambda line: line!=course_data_header)\
.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
course_desc_data = sc.textFile("2006-2016-course-prof.csv")
course_desc_header = course_desc_data.take(1)[0]
course_descs = course_desc_data.filter(lambda line: line!=course_desc_header)\
.map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()

course_titles = course_descs.map(lambda x: (int(x[0]),x[1]))

numPartitions= 4
trainingData = course_grades.filter(lambda x: x[0]< 499523) \
                  .union(course_grades) \
                  .repartition(numPartitions) \
                  .cache()
validationData = course_grades.filter(lambda x:x[0] >=499523 and x[0] < 710353) \
                   .repartition(numPartitions) \
                   .cache()
testData = course_grades.filter(lambda x: x[0]>= 710353).cache()

print('training count %s %s:' %(trainingData.count(), trainingData.take(3)))
print('validation count %s %s:' %(validationData.count(), validationData.take(3)))
print('testData count %s %s:' %(testData.count(), testData.take(3)))
iterations = [10, 20]
regularization_parameter = [0.1]
ranks = [4, 8, 12]

min_error = float('inf')
best_rank = 0
best_iteration = -1
bestLambda = -1
for rank, lmbda, numIter in itertools.product(ranks, regularization_parameter, iterations):
    model = ALS.train(trainingData, rank, numIter, lmbda)
    predictions = model.predictAll(validationData.map(lambda r: (r[0], r[1])))
    print('predictions: %s' %predictions.take(3))
    print('validationData: %s' %validationData.take(3))
    rates_and_preds = predictions.map(lambda r: ((r[0], r[1]),r[2])) \
                         .join(validationData.map(lambda x: ((x[0], x[1]), x[2]))) \
                         .values()
    print('rates_and_preds: %s' %rates_and_preds.take(3))
    error = math.sqrt((rates_and_preds.map(lambda r: (r[0] - r[1])**2)).reduce(add) / float(validationData.count()))
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank
        bestLambda = lmbda
        bestNumIter = numIter
print ('The best model was trained with rank %s , lambda %s, with iterations %d with RMSE for validation data %f' % (best_rank, bestLambda, bestNumIter, min_error))

model = ALS.train(trainingData, best_rank, bestNumIter, bestLambda)
predictions = model.predictAll(testData.map(lambda r: (r[0], r[1])))
rates_and_preds = predictions.map(lambda r: ((r[0], r[1]),r[2])) \
                     .join(testData.map(lambda x: ((x[0], x[1]), x[2]))) \
                     .values()
error = math.sqrt(rates_and_preds.map(lambda r: (r[0] - r[1])**2).reduce(add) / float(testData.count()))

print ('For testing data the RMSE is %s' % (error))

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

course_with_grades = (course_grades.map(lambda x: (x[1], x[2])).groupByKey())
course_with_average_grade = course_with_grades.map(get_counts_and_averages)
course_with_grades_count = course_with_average_grade.map(lambda x: (x[0], x[1][0]))

new_student_id = 800414

# new_grades_model = ALS.train(course_grades, best_rank, bestNumIter, bestLambda)
course_set = (course_grades.filter(lambda x: x[0]== new_student_id)).map(lambda x: x[1])
course_set2 = course_set.collect()
new_student_new_courses = (course_descs.filter(lambda x: x[0] not in course_set2).map(lambda x: (new_student_id, x[0])))

student_course_recommendation_RDD = model.predictAll(new_student_new_courses)

student_course_recommendation_set_RDD = student_course_recommendation_RDD.map(lambda x: (x.product, x.rating))
student_course_count_title_rdd = \
    student_course_recommendation_set_RDD.join(course_titles).join(course_with_grades_count)
print('new set: %s'%student_course_count_title_rdd.take(3))

student_course_count_title_rdd = \
    student_course_count_title_rdd.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
print('title and count: %s'%student_course_count_title_rdd.take(3))

top_courses = student_course_count_title_rdd.filter(lambda r: r[2]>=50).takeOrdered(10, key=lambda x: -x[1])

print ('TOP recommended courses for student 800414 with more than 50 student grades):\n%s' % '\n'.join(map(str, top_courses)))
