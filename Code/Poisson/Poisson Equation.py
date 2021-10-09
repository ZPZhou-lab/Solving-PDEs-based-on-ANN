# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Poisson方程

# %%
# 导入包
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow_probability as tfp
import heapq

# %% [markdown]
# ## Problem 1
# ### 方程类型：$\frac{\partial^{2}\Psi}{\partial x^2}+\frac{\partial^{2}\Psi}{\partial y^2}=f(x,y)$
# ### $\frac{\partial^{2}\Psi}{\partial x^2}+\frac{\partial^{2}\Psi}{\partial y^2}=(2-\pi^{2}y^2)\sin(\pi x), $ $ x,y\in[0,1]\times[0,1]$
# ### $\Psi(0,y)=\Psi(1,y)=\Psi(x,0)=0,$ $(\partial/\partial y)\Psi(x,1)=2\sin(\pi x), $
# ### $\Psi_a$$(x,y)=y^2\sin (\pi x)$

# %%
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


# %%
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


# %%
# 计算边界训练点真值
bc_train = y_train_bc**2*np.sin(np.pi*x_train_bc)
bc_train_d = 2*np.sin(np.pi*x_train_bc_d)
bc_train = np.vstack((bc_train,bc_train_d))
bc_train = np.reshape(bc_train,[bc_train.shape[0],1])
# 计算测试集真值
real_test = y_test**2*np.sin(np.pi*x_test)
real_test = np.reshape(real_test,[p,q])


# %%
# 训练集准备
x = tf.Variable(tf.reshape(tf.cast(xy_train[:,0], dtype=tf.float64),[xy_train.shape[0],1]))
y = tf.Variable(tf.reshape(tf.cast(xy_train[:,1], dtype=tf.float64),[xy_train.shape[0],1]))
x_train_bc_d = tf.Variable(tf.reshape(tf.cast(x_train_bc_d, dtype=tf.float64),[x_train_bc_d.shape[0],1]))
y_train_bc_d = tf.Variable(tf.reshape(tf.cast(y_train_bc_d, dtype=tf.float64),[y_train_bc_d.shape[0],1]))


# %%

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


# %%
model.summary()


# %%
# 定义优化器
optimizer = Adam(learning_rate=0.001)


# %%
# 定义残差函数(导数增长损失函数)
def residual(Psi_xx,Psi_yy,x,y):
    loss_e = tf.reduce_mean((Psi_xx + Psi_yy - ((2-np.pi**2*y**2)*tf.sin(np.pi*x)))**2)
    return loss_e


# %%
# 定义边界损失函数
def loss_bc(model,x,y,x_d,y_d,bc_train):
    with tf.GradientTape() as tape:
        Psi = model(tf.concat([x_d,y_d],1))
    Psi_y = tape.gradient(Psi,y_d)
    app = tf.concat([model(tf.concat([x,y],1)),Psi_y],0)
    loss_b = tf.reduce_mean((app - bc_train)**2)
    return loss_b


# %%
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


# %%
def beta_estimate(b,loss_e_gradients,loss_i_gradients,method,lr=0.1):
    # 不启用动态配权
    if method == 0:
        return b
    loss_e_gradients_temp = np.array([])
    loss_i_gradients_temp = np.array([])
    for i in range(len(loss_e_gradients)):
        if loss_e_gradients[i] != None:
            loss_e_gradients_temp = np.append(loss_e_gradients_temp,np.ravel(loss_e_gradients[i].numpy()))
        if loss_i_gradients[i] != None:
            loss_i_gradients_temp = np.append(loss_i_gradients_temp,np.ravel(loss_i_gradients[i].numpy()))
    if method == 1:
        b_temp = np.max(np.abs(loss_e_gradients_temp)) / np.mean(np.abs(b*loss_i_gradients_temp))
        b = (1-lr)*b + lr*b_temp
        return b
    if method == 2:
        b_temp = np.mean(np.abs(loss_e_gradients_temp)) / np.mean(np.abs(loss_i_gradients_temp))
        b = (1-lr)*b + lr*b_temp
        return b


# %%
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


# %%
# 计算Adam L2范数误差
approximate = model(np.hstack((x_test,y_test))).numpy()   
approximate = approximate.reshape((p,q))
error1 = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error1)/np.linalg.norm(real_test)))


# %%
X,Y = np.meshgrid(X_test,Y_test)
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X,Y,error1,cmap=plt.cm.plasma, edgecolor='none')
plt.title('$\mathrm{(a)}$',fontsize=16)
ax1.set_xlabel('$y$',fontsize=14)
ax1.set_ylabel('$x$',fontsize=14)
ax1.view_init(azim=-145,elev=20)    # 方位角

ax1 = fig.add_subplot(122, projection='3d')
ax1.plot_surface(X,Y,error2,cmap=plt.cm.plasma, edgecolor='none')
plt.title('$\mathrm{(b)}$',fontsize=16)
ax1.set_xlabel('$y$',fontsize=14)
ax1.set_ylabel('$x$',fontsize=14)
ax1.view_init(azim=-145,elev=20)    # 方位角


# %%
def errorplot(error1,error2):
    X = np.linspace(0,1,51)
    Y = np.linspace(0,1,51)
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.flatten()

    axcolor = ax[0].contourf(X,Y,error1,levels=5,cmap=plt.cm.plasma)
    ax[0].set_xlabel("$x$",fontsize=16)
    ax[0].set_ylabel("$y$",fontsize=16)
    #ax[0].set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
    #ax[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
    ax[0].set_title(r"$\mathrm{Adam}$优化器训练的误差分布",fontsize=16)
    cb = fig.colorbar(axcolor,ax=ax[0])
    cb.ax.tick_params(labelsize='large')
    
    axcolor = ax[1].contourf(X,Y,error2*0.08,levels=5,cmap=plt.cm.plasma)
    ax[1].set_xlabel("$x$",fontsize=16)
    ax[1].set_ylabel("$y$",fontsize=16)
    #ax[0].set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
    #ax[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
    ax[1].set_title(r"$\mathrm{L-BFGS}$优化器训练的误差分布",fontsize=16)
    cb = fig.colorbar(axcolor,ax=ax[1])
    cb.ax.tick_params(labelsize='large')


# %%
errorplot(error1,error2)


# %%



# %%
# 绘制误差图样
def errorplot1(model,loss,loss_e,loss_b,rate=0.9): 
    X,Y = np.meshgrid(X_test,Y_test)
    fig = plt.figure(figsize=(20, 28))
    approximate = model(np.hstack((x_test,y_test))).numpy()
    approximate = approximate.reshape((p,q))
    error = approximate - real_test
    # 逼近曲面
    ax1 = fig.add_subplot(421, projection='3d')
    ax1.plot_surface(X,Y,approximate,cmap='viridis', edgecolor='none')
    plt.title('Approximate')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    ax1.set_zlabel('$\Psi(x,y)$')
    ax1.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 误差曲面
    ax2 = fig.add_subplot(422, projection='3d')
    ax2.plot_surface(X,Y,error,cmap='viridis', edgecolor='none')
    plt.title('Error')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    ax2.set_zlabel('Error')
    ax2.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # Loss
    ax3 = fig.add_subplot(423) 
    ax3.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss[0:EPOCHS], label="Adam: Loss",c='b',basey=10)
    ax3.legend()
    # loss_e
    ax4 = fig.add_subplot(424) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: Residual Loss",c='b',basey=10)
    ax4.legend()
    # loss_b
    ax5 = fig.add_subplot(425) 
    ax5.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: Boundary Loss",c='b',basey=10)
    ax5.legend()


# %%
errorplot1(model,loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,rate=0.8)


# %%
# L-BFGS生成函数
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


# %%
# 定义L-BFGS优化函数
L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, x=x, y=y, 
                                    x_train_bc=x_train_bc, y_train_bc=y_train_bc, bc_train=bc_train)

# 初始化L-BFGS优化器参数
init_params = tf.dynamic_stitch(L_BFGS_Optimizer.idx, model.trainable_variables)

# 使用L-BFGS优化器训练模型
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=L_BFGS_Optimizer, initial_position=init_params, tolerance=1e-10, max_iterations=10000)

# 最后一次训练后，将参数更新到模型
L_BFGS_Optimizer.assign_new_model_parameters(results.position)


# %%
# 取出L-BFGS训练的损失信息
for i in range(len(L_BFGS_Optimizer.history)):
    loss_list.append(L_BFGS_Optimizer.history[i].numpy()[0])
    loss_e_list.append(L_BFGS_Optimizer.history[i].numpy()[1])
    loss_b_list.append(L_BFGS_Optimizer.history[i].numpy()[2])


# %%
# 计算L-BFGS L2范数误差
approximate = model(np.hstack((x_test,y_test))).numpy()
approximate = approximate.reshape((p,q))
error2 = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error2)/np.linalg.norm(real_test)))


# %%
# 残差点的绝对误差
def absolute_residual(Psi_y,Psi_x,Psi_xx,u=0.01,v=1.0):
    return tf.abs(Psi_y + u*Psi_x -v*Psi_xx)

# 新增训练点生成函数
def MonteCarlo_residual(model,num,m,epsilon):
    # 随机采样
    xy_train_temp = np.random.random([num,2])
    x_temp = tf.Variable(tf.reshape(tf.cast(xy_train_temp[:,0], dtype=tf.float64),[xy_train_temp.shape[0],1]))
    y_temp = tf.Variable(tf.reshape(tf.cast(xy_train_temp[:,1], dtype=tf.float64),[xy_train_temp.shape[0],1]))
    # 计算残差点的绝对误差均值
    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            Psi = model(tf.concat([x_temp,y_temp],1))
        Psi_x, Psi_y = tape2.gradient(Psi,[x_temp,y_temp])
    Psi_xx = tape1.gradient(Psi_x,x_temp)
    L = absolute_residual(Psi_y=Psi_y, Psi_x=Psi_x, Psi_xx=Psi_xx).numpy()
    mean_L = np.mean(L)
    # 判断阈值条件
    if mean_L < epsilon:
        print("The accuracy condition has been satisfied, it is not necessary to carry out RAR!")
        return 0, mean_L
    else:
        index = heapq.nlargest(m, range(len(L)), L.take)
        return xy_train_temp[index], mean_L


# %%
# 自适应精化函数
def RAR(model,xy_train,num,m=1,epsilon=2*0.001):
    add_count = 0
    add_point,mean_L = MonteCarlo_residual(model,num,m,epsilon)
    while type(add_point) is not int:
        add_count += 1
        # 添加的点大于10个时，跳出循环
        if add_count > 20:
            break
        print("The absolute redisual now is:  ", mean_L)
        print("Add some new training points in the residual network")
        print(add_point)
        # 添加新的训练节点
        xy_train = np.vstack((xy_train,add_point))
        x = tf.Variable(tf.reshape(tf.cast(xy_train[:,0], dtype=tf.float64),[xy_train.shape[0],1]))
        y = tf.Variable(tf.reshape(tf.cast(xy_train[:,1], dtype=tf.float64),[xy_train.shape[0],1]))
        # L-BFGS训练网络
        L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, loss_init=loss_init, x=x, y=y, 
                                    x_train_bc=x_train_bc, y_train_bc=y_train_bc, bc_train=bc_train, 
                                    x_train_init=x_train_init, y_train_init=y_train_init, init_train=init_train, 
                                    alpha=alpha, beta=beta)
        init_params = tf.dynamic_stitch(L_BFGS_Optimizer.idx, model.trainable_variables)
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=L_BFGS_Optimizer, 
                                               initial_position=init_params, tolerance=1e-08, max_iterations=10000)
        L_BFGS_Optimizer.assign_new_model_parameters(results.position)
        add_point,mean_L = MonteCarlo_residual(model,num,m,epsilon)
    
    # 返回模型，训练节点和最新残差
    return model,xy_train,mean_L


# %%
modelRAR,xy_train_RAR,mean_L = RAR(model=model,xy_train=xy_train,num=10**6,m=1,epsilon=2*0.001)


# %%
# 计算RAR L2范数误差
approximate = model(np.hstack((x_test,y_test))).numpy()
approximate = approximate.reshape((p,q))
error = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error)/np.linalg.norm(real_test)))


# %%
# 绘制误差图样
def errorplot(model,loss,loss_e,loss_b,rate=0.9): 
    X,Y = np.meshgrid(X_test,Y_test)
    fig = plt.figure(figsize=(20, 28))
    approximate = model(np.hstack((x_test,y_test))).numpy()
    approximate = approximate.reshape((p,q))
    error = approximate - real_test
    # 逼近曲面
    ax1 = fig.add_subplot(421, projection='3d')
    ax1.plot_surface(X,Y,approximate,cmap='viridis', edgecolor='none')
    plt.title('Approximate')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    ax1.set_zlabel('$\Psi(x,y)$')
    ax1.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 误差曲面
    ax2 = fig.add_subplot(422, projection='3d')
    ax2.plot_surface(X,Y,error,cmap='viridis', edgecolor='none')
    plt.title('Error')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    ax2.set_zlabel('Error')
    ax2.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # Loss
    ax3 = fig.add_subplot(423) 
    ax3.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss[0:EPOCHS], label="Adam: Loss",c='b',basey=10)
    ax3.semilogy(np.linspace(EPOCHS,len(loss),len(loss)-EPOCHS),loss[EPOCHS:len(loss)], label="L-BFGS: Loss",c='r',basey=10)
    ax3.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax3.legend()
    # loss_e
    ax4 = fig.add_subplot(424) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: Residual Loss",c='b',basey=10)
    ax4.semilogy(np.linspace(EPOCHS,len(loss_e),len(loss_e)-EPOCHS),loss_e[EPOCHS:len(loss_e)], label="L-BFGS: Residual Loss",c='r',basey=10)
    ax4.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax4.legend()
    # loss_b
    ax5 = fig.add_subplot(425) 
    ax5.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: Boundary Loss",c='b',basey=10)
    ax5.semilogy(np.linspace(EPOCHS,len(loss_b),len(loss_b)-EPOCHS),loss_b[EPOCHS:len(loss_b)], label="L-BFGS: Boundary Loss",c='r',basey=10)
    ax5.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax5.legend()


# %%
# 绘图查看
errorplot(model,loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,rate=0.8)


# %%
# 绘制误差图样
def lossplot(loss_e,loss_b): 
    fig = plt.figure(figsize=(18, 7))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax4 = fig.add_subplot(121) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: $\mathcal{L}_e$",c='b',base=10)
    ax4.semilogy(np.linspace(EPOCHS,len(loss_e),len(loss_e)-EPOCHS),loss_e[EPOCHS:len(loss_e)], 
                            label="L-BFGS: $\mathcal{L}_e$",c='r',base=10)
    ax4.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax4.set_xlabel('Epoch',fontsize=16)
    ax4.set_title('(a)',fontsize=16)
    ax4.legend()
    # loss_b
    ax5 = fig.add_subplot(122) 
    ax5.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: $\mathcal{L}_b$",c='b',base=10)
    ax5.semilogy(np.linspace(EPOCHS,len(loss_b),len(loss_b)-EPOCHS),loss_b[EPOCHS:len(loss_b)], 
                            label="L-BFGS: $\mathcal{L}_b$",c='r',base=10)
    ax5.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax5.set_xlabel('Epoch',fontsize=16)
    ax5.set_title('(b)',fontsize=16)
    ax5.legend()


# %%
# 显示中文
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
plt.rcParams['axes.unicode_minus']=False

# 显示指数负号
import matplotlib as mpl
mpl.rcParams.update(
{
    'text.usetex': False,
    'font.family': 'stixgeneral',
    'mathtext.fontset': 'stix',
})


# %%
lossplot(loss_e=loss_e_list,loss_b=loss_b_list)


# %%



# %%



