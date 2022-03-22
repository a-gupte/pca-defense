# ! pip install adversarial-robustness-toolbox
# imports and set up
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from google.colab import files
import torchvision.models as models
import numpy as np
from sklearn import decomposition

from art.attacks.evasion import FastGradientMethod, CarliniL2Method, ProjectedGradientDescentPyTorch, DeepFool, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# Model
model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

## Data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

imagenet_dataset = datasets.ImageFolder(
    "./imagenet10-val",
     transforms.Compose([
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            ## I removed the randomness
            transforms.ToTensor(),
            normalize,
        ]))

loader = DataLoader(imagenet_dataset, batch_size=2, shuffle=False)
label_map = [2, 31, 34, 61, 99, 121, 208, 281, 309, 388]

for images, labels in loader:
    print('shape of images: ', images.shape)
    labels = np.array([label_map[i] for i in labels])
    min_pixel_value, max_pixel_value = torch.amin(images), torch.amax(images)
    images = images.detach().cpu().numpy()
    break

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 224, 224),
    nb_classes=1000, # imagenet10, this is true
)

# Step 5: Evaluate the ART classifier on benign test examples

# predictions = classifier.predict(images)
# predictions = np.argmax(predictions, axis=1)
# print('shape of predictions: ', predictions.shape)

# # Step 6: Generate adversarial test examples
# # attack = FastGradientMethod(estimator=classifier, eps=0.5)
# # attack = CarliniL2Method(classifier=classifier)
# # attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=0.05, eps_step = 0.01/12, max_iter=50)
# # attack = DeepFool(classifier=classifier, max_iter=80, epsilon=0.1)
# # JSMA attack with default parameters
# # attack = SaliencyMapMethod(classifier=classifier)

# adversarial_images = attack.generate(x=images)

# # print('attack doing anything?? ', images == adversarial_images) # seems to be doing something, the right thing even!

# print('shape of adversarial_images: ', adversarial_images.shape)

# # Step 7: Evaluate the ART classifier on adversarial test examples

# attacked_predictions = classifier.predict(adversarial_images)
# attacked_predictions = np.argmax(attacked_predictions, axis=1)
# print('attacked_predictions: ', attacked_predictions)

# success_rate = np.sum(predictions != attacked_predictions) / len(labels)
# print("Success rate: {}%".format(success_rate * 100))

# ## Function to compute (k,p) points
# pca = decomposition.PCA()

# def plot_images(X):
#     # X.shape = [rows, columns, channels]
#     channels, rows, columns = X.shape
#     X = np.swapaxes(X, 0, 1)
#     X = np.swapaxes(X, 1, 2)
#     # X = X.detach().cpu().numpy()
#     plt.imshow(X) # takes input of shape [rows, columns, channels]
#     plt.show()
# plot_images(adversarial_images[1])

# path = './attacks/imagenet10-deepfool/imagenet10-deepfool'
# path = './attacks/fgsm-0-03/'
path = './attacks/pgd-0-01/'
done, image_dimension, num_classes = 500, 224, 1000

def image_to_matrix(X):
  # input.shape : [1, channels, rows, columns]
  # output.shape: [rows, columns * channels]
  _, channels, rows, columns = X.shape
  X = np.reshape(X, [channels, rows, columns])
  X = np.swapaxes(X, 0, 1)
  X = np.swapaxes(X, 1, 2)
  X = np.reshape(X, [rows, channels * columns])
  return X

def matrix_to_image(X):
  # input.shape  : [rows, columns * channels]
  # output.shape : [channels, rows, columns]
  rows, columns, channels = 224, 224, 3
  X = np.reshape(X, [rows, columns, channels])
  X = np.swapaxes(X, 1, 2)
  X = np.swapaxes(X, 0, 1)
  return [X]

def plot_image(X):
  X = np.array(X)
  _, channels, rows, columns = X.shape
  print('here', X.shape)
  X = np.reshape(X, [channels, rows, columns])
  X = np.swapaxes(X, 0, 1)
  X = np.swapaxes(X, 1, 2)
  plt.imshow(X)
  plt.show()

def pca(X, k):
  U, S, V = np.linalg.svd(X, full_matrices=False)
  Uk, Sk, Vk_t = U[:, :k], S[:k], V[:k, :]
  Xk = Uk @ np.diag(Sk) @ Vk_t
  return Xk


# k = 5
# for X, y in loader:
#   y = np.array([label_map[i] for i in y])
#   min_pixel_value, max_pixel_value = torch.amin(X), torch.amax(X)
#   X = X.detach().cpu().numpy()
#   plot_image(X)
#   matrix_X = image_to_matrix(X)
#   Xk = pca(matrix_X, k)
#   X = matrix_to_image(Xk)
#   plot_image(X)

#   break

def compute_prediction_set(X, y, plot=False, model=model):
  prediction_list = []
  logits_list = []

  _, channels, rows, columns = X.shape 
  k = min(rows, columns * channels)

  X = image_to_matrix(X)
#   X_mean = np.mean(X, axis=0)
#   print(X_mean.shape)
#   print(X[0, :5])
#   print(X_mean[:5])
#   print((X-X_mean)[0, :5])
#   print((X-X_mean.shape))
#   return
  U, S, V = np.linalg.svd(X, full_matrices=False)

  while k > 0:
    Uk, Sk, Vk_t = U[:, :k], S[:k], V[:k, :]
    Xk = Uk @ np.diag(Sk) @ Vk_t
    Xk = matrix_to_image(Xk) # + X_mean)
    y = model(torch.Tensor(Xk).to(device))

    prediction = y.max(dim=1)[1]
    prediction_list.append(prediction.detach().cpu().numpy()[0])
    logits = np.round_(nn.Softmax()(y)[0].detach().cpu().numpy(), 5)
    logits_list.append(logits)
    k = k-1

  return prediction_list, logits_list

loader = DataLoader(imagenet_dataset, batch_size=1, shuffle=False)

j = 0

all_predictions_adv = np.zeros((done, image_dimension), dtype=int)
all_logits_adv      = np.zeros((done, image_dimension, num_classes))

for X, y in loader:
  print(j)

  y = np.array([label_map[i] for i in y])
  min_pixel_value, max_pixel_value = torch.amin(X), torch.amax(X)
  X = X.detach().cpu().numpy()

  # X_adv = attack.generate(x=X)
  # with open(path + f'adv_images{j}.npy', 'wb') as f:
  #   np.save(f, X_adv[0])
  X_adv = np.load(path + f'adv_images{j}.npy')
  X_adv = np.array([X_adv,])

#   predictions, logits = compute_prediction_set(X, y)
  predictions_adv, logits_adv = compute_prediction_set(X_adv, y)
  
  print(predictions_adv)

#   all_predictions[j] = predictions
  all_predictions_adv[j] = predictions_adv
#   all_logits[j] = logits
  all_logits_adv[j] = logits_adv
  print('\n\n')

  j += 1

# with open(path + f'all_predictions.npy', 'wb') as f:
#   np.save(f, all_predictions)
with open(path + f'all_predictions_adv.npy', 'wb') as f:
  np.save(f, all_predictions_adv)
# with open(path + f'all_logits.npy', 'wb') as f:
#   np.save(f, all_logits)
with open(path + f'all_logits_adv.npy', 'wb') as f:
  np.save(f, all_logits_adv)

import numpy as np
# path = 'attacks/carlini-wagner/'
# path = 'attacks/deepfool-0-01/imagenet10-deepfool'
# path = 'attacks/fgsm-0-03/'

all_predictions = np.load('attacks/all_predictions.npy')
all_predictions_adv = np.load(path + 'all_predictions_adv.npy')
all_logits = np.load('attacks/all_logits.npy')
all_logits_adv = np.load(path + 'all_logits_adv.npy')

image_dimension = 224

# compute and plot the k,p points for each attack. it's okay if it's not super efficient. the major bottelneck was loading.
def k_point(predictions):
  for i, e in enumerate(predictions):
    if e != predictions[0]:
      return image_dimension-i
  return 0

def p_point(predictions, logits, k_point):
  return logits[image_dimension - k_point-1][predictions[image_dimension-k_point-1]]

def compute_kp_points_all_image(all_predictions, all_logits):
  k_points = []
  p_points = []
  for predictions, logits in zip(all_predictions, all_logits):
    k = k_point(predictions)
    p = p_point(predictions, logits, k)
    k_points.append(k)
    p_points.append(p)
  return k_points, p_points

k_adversarial, p_adversarial = compute_kp_points_all_image(all_predictions_adv, all_logits_adv)


with open(path + f'k_adversarial.npy', 'wb') as f:
  np.save(f, k_adversarial)
with open(path + f'p_adversarial.npy', 'wb') as f:
  np.save(f, p_adversarial)

import numpy as np
k_clean, p_clean = np.load('attacks/k_clean.npy'), np.load('attacks/p_clean.npy')
k_adversarial, p_adversarial = np.load(path + 'k_adversarial.npy'), np.load(path + 'p_adversarial.npy')

# rcParams['figure.figsize'] = 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.scatter(k_clean, p_clean, color='black')
plt.scatter(k_adversarial, p_adversarial, color='red')
plt.legend(['Clean images', 'Adversarial images'])
plt.ylabel(r'$p$')
plt.xlabel(r'$x$')
plt.show()

rng = np.random.default_rng()

X_clean = np.vstack((k_clean, p_clean))
# np.random.shuffle(X_clean)
# rng.shuffle(X_clean)
X_adv = np.vstack((k_adversarial, p_adversarial))
# np.random.shuffle(X_adv)
# rng.shuffle(X_adv)

X_clean_train, X_adv_train = X_clean[:, :400], X_adv[:, :400]
X_clean_test, X_adv_test = X_clean[:, 400:], X_adv[:, 400:]

y_clean, y_adv = np.zeros(500), np.ones(500)
y_clean_train, y_adv_train = np.zeros(400), np.ones(400)
y_clean_test, y_adv_test = np.zeros(100), np.ones(100)

X_all = np.hstack((X_clean, X_adv)).T
y_all = np.hstack((y_clean, y_adv))

X_train = np.hstack((X_clean_train, X_adv_train)).T
y_train = np.hstack((y_clean_train, y_adv_train))

X_test = np.hstack((X_clean_test, X_adv_test)).T
y_test = np.hstack((y_clean_test, y_adv_test))

# logistic regression // linear classifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print('training accuracy', clf.score(X_train, y_train))
print('testing accuracy', clf.score(X_test, y_test))

successful_detection_linear = np.array(clf.predict(X_adv.T), dtype=int)
# print(successful_detection)

# Retrieve the model parameters.
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# Plot the data and the classification with the decision boundary.
xmin, xmax = 0, 224
ymin, ymax = 0.0, 1.0
xd = np.array([xmin, xmax])
yd = m*xd + c

plt.scatter(*X_all[y_all==0].T, s=8, #alpha=0.5, 
            color='black')
plt.scatter(*X_all[y_all==1].T, s=8, #alpha=0.5, 
            color='red')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.ylabel(r'$p$')
plt.xlabel(r'$k$')
plt.legend(['Clean images', 'Adversarial images'])
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='black', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='red', alpha=0.2)

plt.show()

## classification with a neural network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(30, 20, 10), random_state=0)
clf.fit(X_train, y_train)
# MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
#               solver='lbfgs')
print('Neural network training accuracy', clf.score(X_train, y_train))
print('Neural network test accuracy', clf.score(X_test, y_test))
# clf.predict(X_adv.T)
successful_detection_nn = np.array(clf.predict(X_adv.T), dtype=int)

# array([1, 0])
# [coef.shape for coef in clf.coefs_]
# [(2, 5), (5, 2), (2, 1)]

def distinct_stretches(l):
    result = list()
    result.append(l[0])
    for e in l:
        if e != result[-1]:
            result.append(e)
    return result

def k_point(l):
    for i, e in enumerate(l):
        if e != l[0]:
            return image_dimension-i

def second_prediction(l):
    for e in l:
        if e != l[0]:
            return e

def defense_accuracy(all_predictions, all_predictions_adv):
  accuracy = 0.0
  num_images = 0
  for predictions_list, predictions_list_adv in zip(all_predictions, all_predictions_adv):
    if predictions_list[0] != predictions_list_adv[0]:
        num_images += 1
        if second_prediction(predictions_list_adv) == predictions_list[0]:
          accuracy += 1.0
  return accuracy/num_images, num_images

def attack_success_rate(all_predictions, all_predictions_adv):
  success = 0.0
  num_images = 0
  for predictions_list, predictions_list_adv in zip(all_predictions, all_predictions_adv):
    num_images += 1
    if predictions_list_adv[0] != predictions_list[0]:
      success += 1.0
  return success/num_images, num_images

print('defense accuracy', defense_accuracy(all_predictions, all_predictions_adv))
print('attack success rate', attack_success_rate(all_predictions, all_predictions_adv))

## conditional on detection
def conditional_defense_accuracy(all_predictions, all_predictions_adv, successful_detection):
    accuracy = 0.0
    num_images = 0
    i=0
    for predictions_list, predictions_list_adv in zip(all_predictions, all_predictions_adv):
        if predictions_list[0] != predictions_list_adv[0] and successful_detection[i] == 1:
            num_images += 1
            if second_prediction(predictions_list_adv) == predictions_list[0]:
              accuracy += 1.0
        i+=1
    return accuracy/num_images, num_images
print('defense accuracy conditioned on successful detection (linear)\n', 
      conditional_defense_accuracy(all_predictions, all_predictions_adv, successful_detection_linear))
print('defense accuracy conditioned on successful detection (nn)\n', 
      conditional_defense_accuracy(all_predictions, all_predictions_adv, successful_detection_nn))

import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
plt.rcParams.update({'font.size': 15})

rcParams['figure.figsize'] = 18, 4
import pickle
with open('./imagenet_label_map.json', 'rb') as fp:
  label_map = pickle.load(fp)

def plot_logits(logits, logits_adv, n=5): # logits.shape = (224, 1000)
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
  plt.title('Logits for top few classes in adversarial image')
  for c in result:
    plt.plot(dims, logits_adv[:, c][::-1])
  plt.legend([label_map[c] for c in result], bbox_to_anchor=(1.05, 1.0), loc='upper left')
  plt.show()
  return result

for i in [0, 50, 100, 150, 200, ]: #250, 300, 350, 400, 450]:
  print(i)
  logits = all_logits[i]
  logits_adv = all_logits_adv[i]
  plot_logits(logits, logits_adv, n=2)


## Checking success rate of the following defense: second highest logit
logits = all_logits[:, 0, :]
logits_adv = all_logits_adv[:, 0, :]
# print(np.argmax(logits, axis=-1))
defense_predictions = np.argpartition(logits_adv.T, -2, axis=0)[-2]
clean_predictions = np.argmax(logits, axis=-1)
adv_predictions = np.argmax(logits_adv, axis=-1)
# print(all(clean_predictions == defense_predictions))
defense_success = np.mean([1 if defense_predictions[i] == clean_predictions[i] and clean_predictions[i] != adv_predictions[i] else 0 for i in range(500)])
print('baseline: second best logit defense success rate', defense_success)



