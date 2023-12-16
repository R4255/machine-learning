# import tensorflow as tf
# def input_fn(features,batch_size=256):
#     #convert the inputs to a dataset without labels
#     return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
# #from_tensor_slices method takes a dictionary of features and creates a dataset with elements that are dictionaries containing the individual feature values
# features=['sepallength','sepalwidth','petallength','petalwidth']
# predict={}
# print("please type numeric values as prompted")
# for feature in features:
#     valid=True
#     while valid:
#         val=input(feature+": ")
#         if not val.isdigit():valid=False
#     predict[feature]=[float(val)]
# predictions=classifier.predict(input_fn=lambda:input_fn(predict))
# for pred_dict in predictions:
#     class_id=pred_dict['class_ids'][0]
#     probability=pred_dict['probabilities'][class_id]
#     print('prediction is "{}"({:.1f}%)'.format(species[class_id],100*probability)