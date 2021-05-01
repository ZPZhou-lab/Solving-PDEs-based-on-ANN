import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow_probability as tfp

# 生成残差训练点
m = 11
n = 11
X_train = np.linspace(0,1,m,endpoint=True) # 生成[0,1]区间m个点
Y_train = np.linspace(0,1,n,endpoint=True) # 生成[0,1]区间n个点
# 生成格点(x_j,y_j)
xy_train = np.zeros((m*n,2),"float64")
for i in range(m):
    for j in range(n):
        xy_train[(i*n+j),0]=X_train[i]
        xy_train[(i*n+j),1]=Y_train[j]     
x_train = np.reshape(xy_train[:,0],[m*n,1])
y_train = np.reshape(xy_train[:,1],[m*n,1])

# 生成边界训练点
n_bc = 11
x0_train = np.hstack((np.zeros([n_bc,1]),np.reshape(np.linspace(0,1,n_bc),[n_bc,1])))
x1_train = np.hstack((np.ones([n_bc,1]),np.reshape(np.linspace(0,1,n_bc),[n_bc,1])))
y0_train = np.hstack((np.reshape(np.linspace(0,1,n_bc),[n_bc,1]),np.zeros([n_bc,1])))
y1_train = np.hstack((np.reshape(np.linspace(0,1,n_bc),[n_bc,1]),np.ones([n_bc,1])))
xy_train_bc = np.vstack((x0_train,x1_train,y0_train))
x_train_bc = xy_train_bc[:,0].reshape(xy_train_bc.shape[0],1)
y_train_bc = xy_train_bc[:,1].reshape(xy_train_bc.shape[0],1)
xy_train_bc_d = y1_train
x_train_bc_d = xy_train_bc_d[:,0].reshape(xy_train_bc_d.shape[0],1)
y_train_bc_d = xy_train_bc_d[:,1].reshape(xy_train_bc_d.shape[0],1)

# 绘制内部训练节点分布
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax1.scatter(xy_train[:,0],xy_train[:,1],s=5)
# 绘制边界训练节点分布
ax2 = fig.add_subplot(132)
ax2.scatter(xy_train_bc[:,0],xy_train_bc[:,1],c=[[1,0.4,0.4]],s=5)
# 将边界添加到内部训练点
ax3 = fig.add_subplot(133)
ax3.scatter(xy_train[:,0],xy_train[:,1],s=5)
ax3.scatter(xy_train_bc[:,0],xy_train_bc[:,1],c=[[1,0.4,0.4]],s=5)

# 生成测试格点
p = 51
q = 51
X_test = np.linspace(0,1,p,endpoint=True) # 生成[0,1]区间p个点
Y_test = np.linspace(0,1,q,endpoint=True) # 生成[0,1]区间q个点
# 生成测试格点(x_j,y_j)
xy_test = np.zeros((p*q,2),"float64")
for i in range(p):
    for j in range(q):
        xy_test[(i*q+j),0]=X_test[i]
        xy_test[(i*q+j),1]=Y_test[j]     
x_test = np.reshape(xy_test[:,0],[p*q,1])
y_test = np.reshape(xy_test[:,1],[p*q,1])

# 计算边界训练点真值
bc_train = y_train_bc**2*np.sin(np.pi*x_train_bc)
bc_train_d = 2*np.sin(np.pi*x_train_bc_d)
bc_train = np.vstack((bc_train,bc_train_d))
bc_train = np.reshape(bc_train,[bc_train.shape[0],1])
# 计算测试集真值
real_test = y_test**2*np.sin(np.pi*x_test)
real_test = np.reshape(real_test,[p,q])

# 训练集准备
x = tf.Variable(tf.reshape(tf.cast(xy_train[:,0], dtype=tf.float64),[xy_train.shape[0],1]))
y = tf.Variable(tf.reshape(tf.cast(xy_train[:,1], dtype=tf.float64),[xy_train.shape[0],1]))
x_train_bc_d = tf.Variable(tf.reshape(tf.cast(x_train_bc_d, dtype=tf.float64),[x_train_bc_d.shape[0],1]))
y_train_bc_d = tf.Variable(tf.reshape(tf.cast(y_train_bc_d, dtype=tf.float64),[y_train_bc_d.shape[0],1]))

# 定义网络结构
struct = [2,30,30,30,1]
tf.keras.backend.set_floatx("float64")
model = keras.models.Sequential()
# 采用Xavier初始化方法
for layer in range(len(struct)-1):
    val = tf.cast(tf.sqrt(6/(struct[layer]+struct[layer+1])),dtype=tf.float64)
    if layer == len(struct)-2:
        model.add(keras.layers.Dense(units=struct[layer+1],kernel_initializer=initializers.random_uniform(-val,val)))
    elif layer == 0:
        model.add(keras.layers.Dense(units=struct[layer+1],input_dim=struct[layer],activation='tanh',
                                     kernel_initializer=initializers.random_uniform(-val,val)))
    else:
        model.add(keras.layers.Dense(units=struct[layer+1],activation='tanh',
                                     kernel_initializer=initializers.random_uniform(-val,val)))
        
model.summary()
# 定义优化器
optimizer = Adam(learning_rate=0.001)


# 定义残差函数(导数增长损失函数)
def residual(Psi_xx,Psi_yy,x,y):
    loss_e = tf.reduce_mean((Psi_xx + Psi_yy - ((2-np.pi**2*y**2)*tf.sin(np.pi*x)))**2)
    return loss_e

# 定义边界损失函数
def loss_bc(model,x,y,x_d,y_d,bc_train):
    with tf.GradientTape() as tape:
        Psi = model(tf.concat([x_d,y_d],1))
    Psi_y = tape.gradient(Psi,y_d)
    app = tf.concat([model(tf.concat([x,y],1)),Psi_y],0)
    loss_b = tf.reduce_mean((app - bc_train)**2)
    return loss_b

# 定义动态权重更新函数
def alpha_estimate(a,loss_e_gradients,loss_b_gradients,method,lr=0.1):
    # 不启用动态配权
    if method == 0:
        return a
    loss_e_gradients_temp = np.array([])
    loss_b_gradients_temp = np.array([])
    for i in range(len(loss_e_gradients)):
        if loss_e_gradients[i] != None:
            loss_e_gradients_temp = np.append(loss_e_gradients_temp,np.ravel(loss_e_gradients[i].numpy()))
        if loss_b_gradients[i] != None:
            loss_b_gradients_temp = np.append(loss_b_gradients_temp,np.ravel(loss_b_gradients[i].numpy()))
    if method == 1:
        a_temp = np.max(np.abs(loss_e_gradients_temp)) / np.mean(np.abs(a*loss_b_gradients_temp))
        a = (1-lr)*a + lr*a_temp
        return a
    if method == 2:
        a_temp = np.mean(np.abs(loss_e_gradients_temp)) / np.mean(np.abs(loss_b_gradients_temp))
        a = (1-lr)*a + lr*a_temp
        return a

# 训练网络
EPOCHS = 30000 # 训练次数
alpha = 1 # 边界权重初始化
loss_list = []
loss_e_list = []
loss_b_list = []
for epoch in range(EPOCHS):
    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape3:
                Psi = model(tf.concat([x,y],1))
            Psi_x, Psi_y = tape3.gradient(Psi,[x,y])
        Psi_xx = tape2.gradient(Psi_x,x)
        Psi_yy = tape2.gradient(Psi_y,y)
        loss_e = residual(Psi_xx=Psi_xx,Psi_yy=Psi_yy,x=x,y=y)
        loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,x_d=x_train_bc_d,y_d=y_train_bc_d,bc_train=bc_train)
        loss = loss_e + alpha*loss_b
    gradients = tape1.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(gradients, model.variables))
    loss_list.append(loss)
    loss_e_list.append(loss_e)
    loss_b_list.append(loss_b)
    if (epoch+1)%5000 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Total loss: {loss}')
        print(f'Residual loss:  {loss_e}')
        print(f'Boundary loss:  {loss_b}')

        
# L-BFGS优化器生成函数
def function_factory(model, residual, loss_bc, x, y, x_train_bc, y_train_bc, bc_train):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """
    
    # 计算model中可训练参数的形状
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)
    
    # 随后将使用tf.dynamic_switch和tf.dynamic_partition来更改形状，下面是准备工作
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices
    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
    part = tf.constant(part)
    
    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))
    
    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
            params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients the `params_1d`.
        """
        # use GradientTape so that we can calculate the gradient of loss
        with tf.GradientTape(persistent=True) as tape1:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape3:
                    Psi = model(tf.concat([x,y],1))
                Psi_x, Psi_y = tape3.gradient(Psi,[x,y])
            Psi_xx = tape2.gradient(Psi_x,x)
            Psi_yy = tape2.gradient(Psi_y,y)
            loss_e = residual(Psi_xx=Psi_xx,Psi_yy=Psi_yy,x=x,y=y)
            loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,x_d=x_train_bc_d,y_d=y_train_bc_d,bc_train=bc_train)
            # calculate the loss
            loss_value = loss_e + loss_b
        # calculate gradients and convert to 1D tf.Tensor
        grads = tape1.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter%2000 == 0:
            tf.print("Iter:", f.iter, "loss:", loss_value)
        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[[loss_value,loss_e,loss_b]], Tout=[])
        return loss_value, grads
    
    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []
    return f

  
# 定义L-BFGS优化函数
L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, x=x, y=y, 
                                    x_train_bc=x_train_bc, y_train_bc=y_train_bc, bc_train=bc_train)

# 初始化L-BFGS优化器参数
init_params = tf.dynamic_stitch(L_BFGS_Optimizer.idx, model.trainable_variables)

# 使用L-BFGS优化器训练模型
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=L_BFGS_Optimizer, initial_position=init_params, tolerance=1e-10, max_iterations=10000)

# 最后一次训练后，将参数更新到模型
L_BFGS_Optimizer.assign_new_model_parameters(results.position)


# 取出L-BFGS训练的损失信息
for i in range(len(L_BFGS_Optimizer.history)):
    loss_list.append(L_BFGS_Optimizer.history[i].numpy()[0])
    loss_e_list.append(L_BFGS_Optimizer.history[i].numpy()[1])
    loss_b_list.append(L_BFGS_Optimizer.history[i].numpy()[2])
    
# 计算L2范数误差
approximate = model(np.hstack((x_test,y_test))).numpy()
approximate = approximate.reshape((p,q))
error2 = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error2)/np.linalg.norm(real_test)))
