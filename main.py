import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import pickle
import json

# gets data in usable form
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# get batch at index from dataset
def get_batch(dict, index, batch_size):
    # shape: (batch_size) 
    labels = dict[b"coarse_labels"][index * batch_size: index * batch_size + batch_size]
    # shape: (batch_size, 32, 32, 3)
    images = dict[b"data"][index * batch_size: index * batch_size + batch_size].reshape(batch_size, 3, 32, 32).transpose(0, 2, 3, 1) #################r

    # convert to numpy array
    labels = np.array(labels)
    images = np.array(images)

    # normalize RGB data
    images = images / 255.0

    return labels, images

# splits image into into patches
def get_patches(image_array, patch_size, color_dim, batch_size):
    patches = np.zeros((batch_size, image_array.shape[1] // patch_size, image_array.shape[2] // patch_size, patch_size, patch_size, color_dim))
    for image_i in range(batch_size):
        for patch_i in range(0, image_array.shape[1] // patch_size):
            for patch_j in range(0, image_array.shape[2] // patch_size):
                for i in range(patch_size):
                    for j in range(patch_size):
                        patches[image_i][patch_i][patch_j][i][j] = image_array[image_i][patch_i * patch_size + i][patch_j * patch_size + j]
    return patches

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

# standard layernorm over last axis
def layernorm(array, gamma, beta):
    mean = np.mean(array, axis = -1, keepdims = True)
    standard_deviation = np.std(array, axis = -1, keepdims = True)
    norm = (array - mean) / (standard_deviation + 1e-5)
    return gamma * norm + beta

def LAYERNORM_BACKWARD(dout, x, gamma, eps=1e-5):
    N = x.shape[-1]
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.sqrt(np.var(x, axis=-1, keepdims=True) + eps)
    rstd = std**-1

    # from karpathy
    w = gamma
   
    norm = (x - mean) * rstd

    db = dout.sum(axis = (0, 1)) 
    dw = (dout * norm).sum(axis = (0, 1))
   
    dnorm = dout * w
    dx = dnorm - dnorm.mean(-1, keepdims=True) - norm * (dnorm * norm).mean(-1, keepdims=True)
    dx *= rstd
    return dx, dw, db

# standard softmax function
def softmax(array):
    exp_array = np.exp(array - np.max(array, axis = -1, keepdims = True))
    return exp_array / np.sum(exp_array, axis = -1, keepdims = True)

def gelu(x):
    return x * (0.5 * (1 + erf(x / np.sqrt(2))))

def gelu_backward(x, dout):
    # exact (derivation in notebook)
    return ( (0.5 * (1 + erf(x / np.sqrt(2)))) + ((x * np.exp(-(x**2 / 2))) / np.sqrt(2 * np.pi)) ) * dout

def cross_entropy(logits, targets):
    # sums all exponentiated logits and applies log, subtracts the target index logit value from entire sum. Derivation of original softmax function to this form in notebook.
    logits_copy = np.copy(logits)
    logits_copy -= np.max(logits_copy, axis = 1, keepdims = True)

    # losses is loss per batch, is shape batch size, 1
    exp = np.exp(logits_copy)
    sum = np.sum(exp, axis = 1, keepdims = True)
    log = np.log(sum)

    target_indices = np.argmax(targets, axis=1)
    batch_indices = np.arange(batch_size) # [1, 2, 3, ..., batch_size - 1]
    target_logits = logits_copy[batch_indices, target_indices] # gets the target logit in each batch
    
    losses = log[:, 0] - target_logits
    loss = np.mean(losses)
    return loss

# displays an image, taken in the form of a numpy array
def visualize(label, image):
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# parameters
rows, cols = 32, 32 # (32, 32) for cifar, (28, 28) for mnist
patch_size = 4
color_dim = 3 # 3 for cifar, 1 for mnist
num_patches = (rows // patch_size) * (cols // patch_size) + 1
embed_dim = 512
ff_dim = 1024
num_heads = 8
num_layers = 6
head_dim = embed_dim // num_heads
num_classes = 20 # 20 for cifar, 10 for mnist
learning_rate = 0.0002
num_epochs = 100
num_examples = 50000 # 50,000 for cifar, 60,000 for mnist
batch_size = 64

# # path to save weights and training stats
checkpoint_weights_path = "saved/checkpoint_model_weights.pkl"
checkpoint_stats_path = "saved/checkpoint_training_stats.json"

# Base: layers: 12, embed_dim: 768, ff_dim: 3072, heads: 12, parameters: 86M
# they did weird patch size upscaling thing, im just going to use size of 4
# batch size: 4096
# uses Adam optimizer, and other training techniques
# fine tune using SGD with momentum
# original Vi-T paper stats:
# cifar 10 got 99.5 accuracy
# cifar 100 got 94 accuracy

# trainable parameters
linear1 = xavier_init(patch_size**2 * color_dim, embed_dim) 
bias1 = np.zeros(embed_dim)
position_embeddings = np.random.randn(num_patches, embed_dim) * 0.01
cls_token = np.random.randn(embed_dim) * 0.01
linear4 = xavier_init(embed_dim, ff_dim)
bias4 = np.zeros(ff_dim)
linear5 = xavier_init(ff_dim, num_classes)
bias5 = np.zeros(num_classes)

# initialize transformer encoder layers' parameters
layers = []
cache = [{} for _ in range(num_layers)]
layer_grads = [{} for _ in range(num_layers)]
for _ in range(num_layers):
    layers.append({
        "gamma1": np.ones(embed_dim),
        "beta1": np.zeros(embed_dim),
        "linear_q": xavier_init(embed_dim, embed_dim),
        "linear_k": xavier_init(embed_dim, embed_dim),
        "linear_v": xavier_init(embed_dim, embed_dim),
        "linear_out": xavier_init(embed_dim, embed_dim),
        "gamma2": np.ones(embed_dim),
        "beta2": np.zeros(embed_dim),
        "linear2": xavier_init(embed_dim, ff_dim),
        "bias2": np.zeros(ff_dim),
        "linear3": xavier_init(ff_dim, embed_dim),
        "bias3": np.zeros(embed_dim)
    })

# for saving and viewing stats
training_stats = {
    "epochs": [],
    "iterations": [],
    "losses": [],
    "accuracies": []
}

# pre-condition
if rows % patch_size != 0 or cols % patch_size != 0:
    print("image and patch size not compatable.")
    quit()

# load data
data = unpickle("cifar-100-python/train")

# main loop
for epoch in range(num_epochs):
    for i in range(num_examples // batch_size):
        # get example
        # label is an int, image is a numpy array of ints of shape 32, 32, 3
        labels, images = get_batch(data, i, batch_size)

        # one hot encode labels, results in shape: (batch_size, num_classes)
        targets = np.stack([np.eye(num_classes)[labels[j]] for j in range(batch_size)])

        # forward pass
        # get patches
        patches = get_patches(images, patch_size, color_dim, batch_size)

        # flatten patches array, then flatten patches themself. Only place using num_patches - 1 becuase it's the only place without the cls token
        patches = patches.reshape(batch_size, num_patches - 1, patch_size * patch_size * color_dim)

        # initial linear matrix multiplication, results in shape: (batch_size, num_patches, embed_dim)
        linear1_out = patches @ linear1

        # add biases, same bias added per patch
        bias1_out = linear1_out + bias1

        # broadcast the cls_token accross batch dimension, results in shape: (batch_size, 1, embed_dim)
        cls_tokens = np.broadcast_to(cls_token, (batch_size, 1, embed_dim))
        cls_token_out = np.concatenate([cls_tokens, bias1_out], axis = 1)

        # add position embeddings
        position_embeddings_out = cls_token_out + position_embeddings

        encoder_input = position_embeddings_out

        # transformer encoder block starts here
        for j in range(num_layers):
            cache[j]["encoder_input"] = encoder_input

            # layernorm
            layernorm1_out = layernorm(encoder_input, layers[j]["gamma1"], layers[j]["beta1"]) 
            cache[j]["layernorm1_out"] = layernorm1_out

            # multi-head attention

            # since this is self attention, q, k, and v are all linear projections of layernorm1_out, results in shape: (batch_size, num_patches, embed_dim)
            q = layernorm1_out @ layers[j]["linear_q"]
            k = layernorm1_out @ layers[j]["linear_k"]
            v = layernorm1_out @ layers[j]["linear_v"]

            # split arrays into head's projections
            q = q.reshape(batch_size, num_patches, num_heads, head_dim)
            k = k.reshape(batch_size, num_patches, num_heads, head_dim)
            v = v.reshape(batch_size, num_patches, num_heads, head_dim)

            # switch dimensions of q, k, and v for operations
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            cache[j]["q"] = q
            cache[j]["k"] = k
            cache[j]["v"] = v

            # matrix multiplication between q and k transposed along last two dimensions, results in shape: (batch_size, num_heads, num_patches, num_patches)
            attention = q @ k.transpose(0, 1, 3, 2)
            cache[j]["attention"] = attention

            # scale by the square root of q, k, and v's dimensions
            scaled_attention = attention / np.sqrt(head_dim)
            cache[j]["scaled_attention"] = scaled_attention

            # apply softmax function to attention
            soft_attention = softmax(scaled_attention)
            cache[j]["soft_attention"] = soft_attention

            # matrix multiplication between attention and k, results in shape: (batch_size, num_heads, num_patches, head_dim)
            attention2 = soft_attention @ v
            cache[j]["attention2"] = attention2

            # concenate heads back into embed dim
            attention_out = attention2.transpose(0, 2, 1, 3).reshape(batch_size, num_patches, embed_dim)
            cache[j]["attention_out"] = attention_out 

            attention_out2 = attention_out @ layers[j]["linear_out"]
            cache[j]["attention_out2"] = attention_out2

            # multi-head attention ends here

            # residual connection
            residual1 = encoder_input + attention_out2
            cache[j]["residual1"] = residual1

            # layernorm
            layernorm2_out = layernorm(residual1, layers[j]["gamma2"], layers[j]["beta2"])
            cache[j]["layernorm2_out"] = layernorm2_out

            # fist MLP, results in shape: (batch_size, num_patches, embed_dim)
            linear2_out = layernorm2_out @ layers[j]["linear2"]
            bias2_out = linear2_out + layers[j]["bias2"]
            gelu1_out = gelu(bias2_out)
            linear3_out = gelu1_out @ layers[j]["linear3"]
            bias3_out = linear3_out + layers[j]["bias3"]
            cache[j]["linear2_out"] = linear2_out
            cache[j]["bias2_out"] = bias2_out
            cache[j]["gelu1_out"] = gelu1_out
            cache[j]["linear3_out"] = linear3_out
            cache[j]["bias3_out"] = bias3_out

            # residual connection
            residual2 = residual1 + bias3_out
            cache[j]["residual2"] = residual2

            encoder_input = residual2

        # transformer block ends here, repeat N times using previous output as input

        # final MLP, results in shape: (batch_size, num_patches, num_classes)
        linear4_out = residual2 @ linear4
        bias4_out = linear4_out + bias4
        linear5_out = bias4_out @ linear5
        bias5_out = linear5_out + bias5
        
        # select cls_token from each batch as logits
        logits = bias5_out[:, 0]

        loss = cross_entropy(logits, targets)
        # end forward pass

        # print while training
        print(f"loss {i}: {loss}")
        predictions = np.argmax(logits, axis=1)
        targets_indices = np.argmax(targets, axis=1)
        correct_predictions = np.sum(predictions == targets_indices)
        accuracy = correct_predictions * 100 / batch_size
        print("accuracy: " + str(accuracy) + "%")

        # log stats
        training_stats["epochs"].append(epoch)
        training_stats["iterations"].append(i)
        training_stats["losses"].append(loss)
        training_stats["accuracies"].append(accuracy)

        # backpropagation

        # cross entropy gradient
        # derivation in notebook
        dlogits = (softmax(logits) - targets) / batch_size 

        # final MLP gradient
        dbias5_out = np.zeros_like(bias5_out)
        dbias5_out[:, 0] = dlogits

        # matrix addition gradient
        # derivation in notebook
        dlinear5_out = dbias5_out
        dbias5 = np.sum(dbias5_out, axis = 1)

        # matrix multiplication gradient
        # derivation in notebook
        dbias4_out = dlinear5_out @ linear5.T
        dlinear5 = bias4_out.transpose(0, 2, 1) @ dlinear5_out

        # matrix addition gradient
        dlinear4_out = dbias4_out
        dbias4 = np.sum(dbias4_out, axis = 1)

        # matrix multiplication gradient
        dresidual2 = dlinear4_out @ linear4.T
        dlinear4 = residual2.transpose(0, 2, 1) @ dlinear4_out

        # NEW transformer block gradient
        dencoder_input = dresidual2
        for j in reversed(range(num_layers)):
            dresidual1 = dencoder_input
            dbias3_out = dencoder_input
            
            # first MLP gradient
            # matrix addition gradient
            dlinear3_out = dbias3_out
            dbias3 = np.sum(dbias3_out, axis = 1)

            # matrix multiplication gradient
            dgelu1_out = dlinear3_out @ layers[j]["linear3"].T
            dlinear3 = cache[j]["gelu1_out"].transpose(0, 2, 1) @ dlinear3_out

            # gelu gradient
            dbias2_out = gelu_backward(cache[j]["bias2_out"], dgelu1_out)

            # matrix addition gradient
            dlinear2_out = dbias2_out
            dbias2 = np.sum(dbias2_out, axis = 1)

            # matrix multiplication gradient
            dlayernorm2_out = dlinear2_out @ layers[j]["linear2"].T
            dlinear2 = cache[j]["layernorm2_out"].transpose(0, 2, 1) @ dlinear2_out

            # layernorm gradient
            temp_dresidual1, dgamma2, dbeta2 = LAYERNORM_BACKWARD(dlayernorm2_out, cache[j]["residual1"], layers[j]["gamma2"])
            dresidual1 += temp_dresidual1

            # residual connection gradient
            dattention_out2 = dresidual1
            
            # matrix multiplication gradient
            dattention_out = dattention_out2 @ layers[j]["linear_out"].T
            dlinear_out = cache[j]["attention_out"].transpose(0, 2, 1) @ dattention_out2

            # transpose gradient
            dattention2 = dattention_out.reshape(batch_size, num_patches, num_heads, head_dim).transpose(0, 2, 1, 3)
            
            # matrix multiplication gradient
            dsoft_attention = dattention2 @ cache[j]["v"].transpose(0, 1, 3, 2)
            dv = cache[j]["soft_attention"].transpose(0, 1, 3, 2) @ dattention2 # not what response has, but I think it's right

            # softmax gradient
            dot = np.sum(dsoft_attention * cache[j]["soft_attention"], axis=-1, keepdims=True)
            dscaled_attention = cache[j]["soft_attention"] * (dsoft_attention - dot)

            # gradient of scalar multiplication
            dattention = (np.sqrt(head_dim)**-1) * dscaled_attention
            
            # matrix multiplication gradient and transpose gradient (transposes on k cancelled out)
            dq = dattention @ cache[j]["k"]
            dk = (cache[j]["q"].transpose(0, 1, 3, 2) @ dattention).transpose(0, 1, 3, 2)

            # transpose gradient
            dq = dq.transpose(0, 2, 1, 3)
            dk = dk.transpose(0, 2, 1, 3)
            dv = dv.transpose(0, 2, 1, 3)

            # reshape gradient
            dq = dq.reshape(batch_size, num_patches, embed_dim)
            dk = dk.reshape(batch_size, num_patches, embed_dim)
            dv = dv.reshape(batch_size, num_patches, embed_dim)

            # matrix multiplication gradient
            dlayernorm1_out = dq @ layers[j]["linear_q"].T
            dlinear_q = cache[j]["layernorm1_out"].transpose(0, 2, 1) @ dq

            dlayernorm1_out += dk @ layers[j]["linear_k"].T
            dlinear_k = cache[j]["layernorm1_out"].transpose(0, 2, 1) @ dk

            dlayernorm1_out += dv @ layers[j]["linear_v"].T
            dlinear_v = cache[j]["layernorm1_out"].transpose(0, 2, 1) @ dv

            # layernorm gradient
            temp_dencoder_input, dgamma1, dbeta1 = LAYERNORM_BACKWARD(dlayernorm1_out, cache[j]["encoder_input"], layers[j]["gamma1"])
            dencoder_input = temp_dencoder_input + dresidual1

            # save gradients 
            layer_grads[j]["dlinear_q"] = dlinear_q
            layer_grads[j]["dlinear_k"] = dlinear_k
            layer_grads[j]["dlinear_v"] = dlinear_v
            layer_grads[j]["dlinear_out"] = dlinear_out
            layer_grads[j]["dgamma1"] = dgamma1
            layer_grads[j]["dbeta1"] = dbeta1
            layer_grads[j]["dgamma2"] = dgamma2
            layer_grads[j]["dbeta2"]  = dbeta2
            layer_grads[j]["dlinear2"] = dlinear2
            layer_grads[j]["dbias2"] = dbias2
            layer_grads[j]["dlinear3"] = dlinear3
            layer_grads[j]["dbias3"] = dbias3

        dposition_embeddings_out = dencoder_input

        # matrix addition gradient
        dcls_token_out = dposition_embeddings_out
        dposition_embeddings = dposition_embeddings_out

        # concatenation gradient
        dcls_token = dcls_token_out[:, 0]
        dbias1_out = dcls_token_out[:, 1:]

        # matrix addition gradient
        dlinear1_out = dbias1_out
        dbias1 = np.sum(dbias1_out, axis = 1)

        # matrix multiplication gradient
        # info: only linear1 is required to calculate, as it's the final trainable parameter in the gradient chain.
        dlinear1 = patches.transpose(0, 2, 1) @ dlinear1_out 
        # end backpropagation

        # optimization using SGD
        linear1 -= np.sum(dlinear1, axis = 0) * learning_rate # was np.mean(), probably shouldn't have been
        bias1 -=  np.sum(dbias1, axis = 0) * learning_rate
        position_embeddings -= np.sum(dposition_embeddings, axis = 0) * learning_rate
        cls_token -= np.sum(dcls_token, axis = 0) * learning_rate
        for j in range(num_layers):
            layers[j]["linear_q"] -=  np.sum(layer_grads[j]["dlinear_q"], axis = 0) * learning_rate
            layers[j]["linear_k"] -=  np.sum(layer_grads[j]["dlinear_k"], axis = 0) * learning_rate
            layers[j]["linear_v"] -=  np.sum(layer_grads[j]["dlinear_v"], axis = 0) * learning_rate
            layers[j]["linear_out"] -=  np.sum(layer_grads[j]["dlinear_out"], axis = 0) * learning_rate
            layers[j]["gamma1"] -= (layer_grads[j]["dgamma1"] / (batch_size * num_patches)) * learning_rate
            layers[j]["beta1"] -= (layer_grads[j]["dbeta1"] / (batch_size * num_patches)) * learning_rate
            layers[j]["gamma2"] -= (layer_grads[j]["dgamma2"] / (batch_size * num_patches)) * learning_rate
            layers[j]["beta2"] -= (layer_grads[j]["dbeta2"] / (batch_size * num_patches)) * learning_rate
            layers[j]["linear2"] -= np.sum(layer_grads[j]["dlinear2"], axis = 0) * learning_rate
            layers[j]["bias2"] -= np.sum(layer_grads[j]["dbias2"], axis = 0) * learning_rate
            layers[j]["linear3"] -= np.sum(layer_grads[j]["dlinear3"], axis = 0) * learning_rate
            layers[j]["bias3"] -=  np.sum(layer_grads[j]["dbias3"], axis = 0) * learning_rate
        linear4 -= np.sum(dlinear4, axis = 0) * learning_rate
        bias4 -= np.sum(dbias4, axis = 0) * learning_rate
        linear5 -= np.sum(dlinear5, axis = 0) * learning_rate
        bias5 -= np.sum(dbias5, axis = 0) * learning_rate

        # after 1 epoch, the model achieves approximately 92% accuracy on mnist, showing the model works.
    
        # after every 100 iterations, save weights, store weights, and store stats
        if i % 100 == 0:
            model_weights = {
                "linear1": linear1,
                "bias1": bias1,
                "position_embeddings": position_embeddings,
                "cls_token": cls_token,
                "linear4": linear4,
                "bias4": bias4,
                "linear5": linear5,
                "bias5": bias5,
                "layers": layers
            }

            with open(checkpoint_weights_path, "wb") as f:
                pickle.dump(model_weights, f)
            
            with open(checkpoint_stats_path, "w") as f:
                json.dump(training_stats, f)
