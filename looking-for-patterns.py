# path = 'drive/MyDrive/imagenet10-pgd-0-05/imagenet10-pgd-0-05'
path = './data/fgsm-0-1/'
import numpy as np
all_predictions = np.load(path + 'all_predictions.npy')
all_predictions_adv = np.load(path + 'all_predictions_adv.npy')
all_logits = np.load(path + 'all_logits.npy')
all_logits_adv = np.load(path + 'all_logits_adv.npy')



# k_clean = np.load(path + 'k_clean.npy')
# k_adversarial = np.load(path + 'k_adversarial.npy')
# p_clean = np.load(path + 'p_clean.npy')
# p_adversarial = np.load(path + 'p_adversarial.npy')


# def distinct_stretches(l):
#     result = list()
#     result.append(l[0])
#     for e in l:
#         if e != result[-1]:
#             result.append(e)
#     return result

def k_point(prediction_list):
	for i, e in enumerate(prediction_list):
		if e != prediction_list[0]:
			return image_dimension - i

# def second_prediction(l):
#     for e in l:
#         if e != l[0]:
#             return e

# def defense_accuracy(all_predictions, all_predictions_adv):
#   accuracy = 0.0
#   num_images = 0
#   for predictions_list, predictions_list_adv in zip(all_predictions, all_predictions_adv):
#     if predictions_list[0] != predictions_list_adv[0]:
#         num_images += 1
#         if second_prediction(predictions_list_adv) == predictions_list[0]:
#           accuracy += 1.0
#   return accuracy/num_images, num_images

# def attack_success_rate(all_predictions, all_predictions_adv):
#   success = 0.0
#   num_images = 0
#   for predictions_list, predictions_list_adv in zip(all_predictions, all_predictions_adv):
#     num_images += 1
#     if predictions_list_adv[0] != predictions_list[0]:
#       success += 1.0
#   return success/num_images, num_images

# print('defense accuracy', defense_accuracy(all_predictions, all_predictions_adv))
# print('attack success rate', attack_success_rate(all_predictions, all_predictions_adv))

# ## conditional on detection
# def conditional_defense_accuracy(all_predictions, all_predictions_adv, successful_detection):
#     accuracy = 0.0
#     num_images = 0
#     i=0
#     for predictions_list, predictions_list_adv in zip(all_predictions, all_predictions_adv):
#         if predictions_list[0] != predictions_list_adv[0] and successful_detection[i] == 1:
#             num_images += 1
#             if second_prediction(predictions_list_adv) == predictions_list[0]:
#               accuracy += 1.0
#         i+=1
#     return accuracy/num_images, num_images
# print('defense accuracy conditioned on successful detection (linear)\n', 
#       conditional_defense_accuracy(all_predictions, all_predictions_adv, successful_detection_linear))
# print('defense accuracy conditioned on successful detection (nn)\n', 
#       conditional_defense_accuracy(all_predictions, all_predictions_adv, successful_detection_nn))

# import matplotlib.pyplot as plt
# from pylab import rcParams
# import numpy as np
# plt.rcParams.update({'font.size': 15})

# rcParams['figure.figsize'] = 18, 4
# import pickle
# with open('drive/MyDrive/imagenet_label_map.json', 'rb') as fp:
#   label_map = pickle.load(fp)

# # for i in [0, 50, 100, 150, 200, ]: #250, 300, 350, 400, 450]:
# #   print(i)
# #   logits = all_logits[i]
# #   logits_adv = all_logits_adv[i]
# #   plot_logits(logits, logits_adv, n=2)

# ## Checking success rate of the following defense: second highest logit
# logits = all_logits[:, 0, :]
# logits_adv = all_logits_adv[:, 0, :]
# # print(np.argmax(logits, axis=-1))
# defense_predictions = np.argpartition(logits_adv.T, -2, axis=0)[-2]
# clean_predictions = np.argmax(logits, axis=-1)
# adv_predictions = np.argmax(logits_adv, axis=-1)
# # print(all(clean_predictions == defense_predictions))
# defense_success = np.mean([1 if defense_predictions[i] == clean_predictions[i] and clean_predictions[i] != adv_predictions[i] else 0 for i in range(500)])
# print('baseline: second best logit defense success rate', defense_success)


# import matplotlib.pyplot as plt
# import numpy as np
# plt.rcParams['text.usetex'] = True
# base_path = 'drive/MyDrive/imagenet10-'
# paths = ['pgd-0-05/imagenet10-pgd-0-05', 'fgsm-0-03/', 'pgd-0-01/', 'deepfool-0-01/', 'carlini-wagner/'
#          'attacks/carlini-wagner/',]
# paths = [base_path + path for path in paths]
# titles = ['Clean images', r'FGSM \epsilon = 0.03', r'PGD \epsilon = 0.01', r'DeepFool \epsilon = 0.01',
#          'Carlini-Wagner']

# from pylab import rcParams
# import numpy as np
# plt.rcParams.update({'font.size': 15})



# rcParams['figure.figsize'] = 25, 25
# import pickle
# with open('drive/MyDrive/imagenet_label_map.json', 'rb') as fp:
#     label_map = pickle.load(fp)
	
# xmin, xmax = 0, 224
# ymin, ymax = 0.0, 1.0
# image_number = 0
	
# top_classes = []
# n=5

# for i, path in enumerate(paths):
#     is_adv = '_adv' if path != 'attacks/' else ''
#     all_logits = np.load(path + f'all_logits{is_adv}.npy')
#     logits = all_logits[image_number]
#     for k in range(224):
#         logits_k = logits[k]
#         top_n_classes = np.argpartition(logits_k, -n)[-n:][::-1]
#         for c in top_n_classes:
#             if c not in top_classes:
#                 top_classes.append(c)
# for i, path in enumerate(paths):
#     is_adv = '_adv' if path != 'attacks/' else ''
#     all_logits = np.load(path + f'all_logits{is_adv}.npy')
#     logits = all_logits[image_number]
#     dims = list(range(224))
#     # plots
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#     plt.xlabel(r'$k$')
#     plt.ylabel('Logits')
#     print(path)
#     plt.subplot(3, 2, i+1)
#     plt.title(titles[i])
#     for c in top_classes:
#         plt.plot(dims, logits[:, c][::-1]) #, color=colors[(c%37) * 3])
	
# plt.legend([label_map[c] for c in top_classes], bbox_to_anchor=(1.5, 1.0), loc='upper left', ncol=2)
# plt.show()
import matplotlib.pyplot as plt

import pickle
with open('./imagenet_label_map.json', 'rb') as fp:
	label_map = pickle.load(fp)

def plot_logits(logits, logits_adv, n=5):
	# logits.shape = (224, 1000)
	result = []
	for k in range(224):
	# clean logits
		logits_k = logits[k]
		top_n_classes = np.argpartition(logits_k, -n)[-n:][::-1]
		for c in top_n_classes:
			if c not in result:
				result.append(c)
		# adv logits
		logits_k = logits_adv[k]
		top_n_classes = np.argpartition(logits_k, -n)[-n:][::-1]
		for c in top_n_classes:
			if c not in result:
				result.append(c)
	dims = list(range(224))
	# clean plot
	plt.subplot(1, 2, 1)
	xmin, xmax = 0, 224
	ymin, ymax = 0.0, 1.0
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)

	plt.xlabel(r'$k$')
	plt.ylabel('Logits')
	plt.title('Logits for top few classes in clean image')
	for c in result:
		plt.plot(dims, logits[:, c][::-1])
	# adversarial plot
	plt.subplot(1, 2, 2)
	plt.xlabel(r'$k$')
	plt.ylabel('Logits')
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	plt.title('Logits for top few classes in adversarial image')
	for c in result:
		plt.plot(dims, logits_adv[:, c][::-1])
	plt.legend([label_map[c] for c in result], bbox_to_anchor=(1.05, 1.0), loc='upper left')
	plt.show()
	return [label_map[c] for c in result]

for i in range(4):
	print(i)
	logits = all_logits[i]
	logits_adv = all_logits_adv[i]
	print(plot_logits(logits, logits_adv))
