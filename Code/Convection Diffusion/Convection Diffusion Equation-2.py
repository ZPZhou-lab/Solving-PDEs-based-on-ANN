# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 对流扩散方程

# %%
# 导入包
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
#coding:utf-8
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


# %%
import matplotlib as mpl
mpl.rcParams.update(
{
    'text.usetex': False,
    'font.family': 'stixgeneral',
    'mathtext.fontset': 'stix',
}
)


# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow_probability as tfp
import heapq

# %% [markdown]
# ## Problem 2
# ### 方程类型：$\phi_t + u\cdot\phi_x + v\cdot \phi_y - (\phi_{xx}+\phi_{yy}) = f(x,y,t)$
# ### 其中，$x,y\in[0,1]$, $t\in[0,+\infty)$, $u(x,y,t)=tDx$, $v(x,y,t)=-tDy$ 
# ### $f(x,y,t)=xye^{x+y}[(1-x)(1-y)+t^2D(y-x)(y+x-xy)+2t(3-x-y-xy)]$
# ### 解析解$\phi=txy(1-x)(1-y)e^{x+y}$
# ### 由解析解给出下面的边界条件：
# $\phi(0,y,t)=\phi(1,y,t)=\phi(x,0,t)=\phi(x,1,t)=0$, $\phi(x,y,0)=0$

# %%
# 生成残差训练点
n_use = 21
t_use = 51
m = n_use
n = n_use
l = t_use
X_train = np.linspace(0,1,m,endpoint=True) # 生成[0,1]区间m个点
Y_train = np.linspace(0,1,n,endpoint=True) # 生成[0,1]区间n个点
T_train = np.linspace(0,0.5,l,endpoint=True) # 生成[0,1]区间l个点
# 生成格点(x_j,y_j,t_j)
xyt_train = np.zeros((m*n*l,3),"float64")
temp = 0
for i in range(m):
    for j in range(n):
        for k in range(l):
            xyt_train[temp,0]=X_train[i]
            xyt_train[temp,1]=Y_train[j]
            xyt_train[temp,2]=T_train[k]
            temp += 1
x_train = np.reshape(xyt_train[:,0],[m*n*l,1])
y_train = np.reshape(xyt_train[:,1],[m*n*l,1])
t_train = np.reshape(xyt_train[:,2],[m*n*l,1])

# 生成边界训练点
x0_train = xyt_train[xyt_train[:,0]==0]
x1_train = xyt_train[xyt_train[:,0]==1]
y0_train = xyt_train[xyt_train[:,1]==0]
y1_train = xyt_train[xyt_train[:,1]==1]
xyt_train_bc = np.vstack((x0_train,x1_train,y0_train,y1_train))
x_train_bc = xyt_train_bc[:,0].reshape(xyt_train_bc.shape[0],1)
y_train_bc = xyt_train_bc[:,1].reshape(xyt_train_bc.shape[0],1)
t_train_bc = xyt_train_bc[:,2].reshape(xyt_train_bc.shape[0],1)

# 生成初值训练点
t0_train = xyt_train[xyt_train[:,2]==0]
x_train_ic = t0_train[:,0].reshape(t0_train.shape[0],1)
y_train_ic = t0_train[:,1].reshape(t0_train.shape[0],1)
t_train_ic = t0_train[:,2].reshape(t0_train.shape[0],1)


# %%
# 生成测试格点
p = 51
q = 51
r = 101
X_test = np.linspace(0,1,p,endpoint=True) # 生成[0,1]区间p个点
Y_test = np.linspace(0,1,q,endpoint=True) # 生成[0,1]区间q个点
T_test = np.linspace(0,0.5,r,endpoint=True) # 生成[0,1]区间r个点
# 生成测试格点(x_j,y_j,t_j)
xyt_test = np.zeros((p*q*r,3),"float64")
temp = 0
for i in range(p):
    for j in range(q):
        for k in range(r):
            xyt_test[temp,0]=X_test[i]
            xyt_test[temp,1]=Y_test[j]     
            xyt_test[temp,2]=T_test[k]
            temp += 1
x_test = np.reshape(xyt_test[:,0],[p*q*r,1])
y_test = np.reshape(xyt_test[:,1],[p*q*r,1])
t_test = np.reshape(xyt_test[:,2],[p*q*r,1])


# %%
# 计算边界训练点真值
bc_train = t_train_bc*x_train_bc*y_train_bc*(1-x_train_bc)*(1-y_train_bc)*np.exp(x_train_bc+y_train_bc)
bc_train = np.reshape(bc_train,[bc_train.shape[0],1])
ic_train = t_train_ic*x_train_ic*y_train_ic*(1-x_train_ic)*(1-y_train_ic)*np.exp(x_train_ic+y_train_ic)
ic_train = np.reshape(ic_train,[ic_train.shape[0],1])
# 计算测试集真值
real_test = t_test*x_test*y_test*(1-x_test)*(1-y_test)*np.exp(x_test+y_test)
real_test = np.reshape(real_test,[p,q,r])


# %%
# 训练集准备
x = tf.Variable(tf.reshape(tf.cast(xyt_train[:,0], dtype=tf.float64),[xyt_train.shape[0],1]))
y = tf.Variable(tf.reshape(tf.cast(xyt_train[:,1], dtype=tf.float64),[xyt_train.shape[0],1]))
t = tf.Variable(tf.reshape(tf.cast(xyt_train[:,2], dtype=tf.float64),[xyt_train.shape[0],1]))


# %%
# 定义网络
struct = [3,50,50,50,50,1]
# 定义优化器
optimizer = Adam(learning_rate=0.001)
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
# 定义残差函数(导数增长损失函数)
def residual(Psi_xx,Psi_yy,Psi_x,Psi_y,Psi_t,x,y,t,D):
    loss_e = tf.reduce_mean((Psi_t+(t*D*x)*Psi_x-(t*D*y)*Psi_y-(Psi_xx+Psi_yy)
                            -(x*y*tf.exp(x+y)*((1-x)*(1-y)+t**2*D*(y-x)*(y+x-x*y)+2*t*(3-x-y-x*y))))**2)
    return loss_e

# 定义边界损失函数
def loss_bc(model,x,y,t,bc_train):
    loss_b = tf.reduce_mean((model(tf.concat([x,y,t],1)) - bc_train)**2)
    return loss_b

def loss_ic(model,x,y,t,ic_train):
    loss_i = tf.reduce_mean((model(tf.concat([x,y,t],1)) - ic_train)**2)
    return loss_i 

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
# L-BFGS生成函数
def function_factory(model, residual, loss_bc, loss_ic, x, y, t, 
                    x_train_bc, y_train_bc, t_train_bc, bc_train,
                    x_train_ic, y_train_ic, t_train_ic, ic_train, alpha, beta, D):
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
                    Psi = model(tf.concat([x,y,t],1))
                Psi_x, Psi_y, Psi_t = tape3.gradient(Psi,[x,y,t])
            Psi_xx = tape2.gradient(Psi_x,x)
            Psi_yy = tape2.gradient(Psi_y,y)
            loss_e = residual(Psi_xx=Psi_xx,Psi_yy=Psi_yy,Psi_x=Psi_x,Psi_y=Psi_y,Psi_t=Psi_t,x=x,y=y,t=t,D=D)
            loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,t=t_train_bc,bc_train=bc_train)
            loss_i = loss_ic(model=model,x=x_train_ic,y=y_train_ic,t=t_train_ic,ic_train=ic_train)
            # calculate the loss
            loss_value = loss_e + alpha*loss_b + beta*loss_i
        # calculate gradients and convert to 1D tf.Tensor
        grads = tape1.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter%2000 == 0:
            tf.print("Iter:", f.iter, "loss:", loss_value)
        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[[loss_value,loss_e,loss_b,loss_i]], Tout=[])
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
EPOCHS = 30000 # 训练次数
alpha = 100 # 边界权重初始化
beta = 100 # 初值权重初始化
D = 10
method = 1
loss_list = []
loss_e_list = []
loss_b_list = []
loss_i_list = []
alpha_list = [alpha]
beta_list = [beta]
for epoch in range(EPOCHS):
    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape3:
                Psi = model(tf.concat([x,y,t],1))
            Psi_x, Psi_y, Psi_t = tape3.gradient(Psi,[x,y,t])
        Psi_xx = tape2.gradient(Psi_x,x)
        Psi_yy = tape2.gradient(Psi_y,y)
        loss_e = residual(Psi_xx=Psi_xx,Psi_yy=Psi_yy,Psi_x=Psi_x,Psi_y=Psi_y,Psi_t=Psi_t,x=x,y=y,t=t,D=D)
        loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,t=t_train_bc,bc_train=bc_train)
        loss_i = loss_ic(model=model,x=x_train_ic,y=y_train_ic,t=t_train_ic,ic_train=ic_train)
        loss = loss_e + alpha*loss_b + beta*loss_i
    if ((epoch+1)%100 == 0) and (method != 0): # 每100次更新一次权重alpha
        loss_e_gradients = tape1.gradient(loss_e,model.variables)
        loss_b_gradients = tape1.gradient(loss_b,model.variables)
        loss_i_gradients = tape1.gradient(loss_i,model.variables)
        alpha = alpha_estimate(a=alpha,loss_e_gradients=loss_e_gradients,loss_b_gradients=loss_b_gradients,method=method)
        beta = beta_estimate(b=beta,loss_e_gradients=loss_e_gradients,loss_i_gradients=loss_i_gradients,method=method)
        alpha_list.append(alpha)
        beta_list.append(beta)
    gradients = tape1.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(gradients, model.variables))
    loss_list.append(loss)
    loss_e_list.append(loss_e)
    loss_b_list.append(loss_b)
    loss_i_list.append(loss_i)
    if (epoch+1)%5000 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Total loss: {loss}')
        print(f'Residual loss:  {loss_e}')
        print(f'Boundary loss:  {loss_b}')
        print(f'Initial loss:  {loss_i}')


# %%
# 计算Adam L2范数误差
approximate = model(np.hstack((x_test,y_test,t_test))).numpy()   
approximate = approximate.reshape((p,q,r))
error = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error)/np.linalg.norm(real_test)))
print("MAE: {}".format(np.max(np.abs(error))))
print("MSE: {}".format(np.mean(np.abs(error))))


# %%
# 绘制误差图样
def lossplot(loss,loss_e,loss_b,loss_i,alpha,beta): 
    fig = plt.figure(figsize=(20, 21))
    # Loss
    ax1 = fig.add_subplot(321) 
    ax1.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss[0:EPOCHS], label="Adam: Loss",c='b',basey=10)
    ax1.legend()
    # loss_e
    ax2 = fig.add_subplot(322) 
    ax2.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: Residual Loss",c='b',basey=10)
    ax2.legend()
    # loss_b
    ax3 = fig.add_subplot(323) 
    ax3.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: Boundary Loss",c='b',basey=10)
    ax3.legend()
    # loss_i
    ax4 = fig.add_subplot(324) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_i[0:EPOCHS], label="Adam: Initial Loss",c='b',basey=10)
    ax4.legend()
    # alpha,beta
    ax5 = fig.add_subplot(325)
    ax5.plot(np.linspace(1,len(alpha),len(alpha)),alpha,label="Alpha",c='b')
    ax5.plot(np.linspace(1,len(beta),len(beta)),beta,label="Beta",c='r')
    ax5.legend()


# %%
lossplot(loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,loss_i=loss_i_list,alpha=alpha_list,beta=beta_list)


# %%
def errorplot(model):
    approximate = model(np.hstack((x_test,y_test,t_test))).numpy()   
    approximate = approximate.reshape((p,q,r))
    error = approximate - real_test
    vmin = np.min(np.abs(error))
    vmax = np.max(np.abs(error))
    Time = ['0.05','0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45']
    X = np.linspace(0,1,51)
    Y = np.linspace(0,1,51)
    fig, ax = plt.subplots(3,3, figsize=(24,21))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.flatten()
    for i in range(9):
        axcolor = ax[i].contourf(X,Y,error[:,:,(i+1)*10],levels=5,cmap=plt.cm.plasma)
        ax[i].set_xlabel("x",fontsize=16)
        ax[i].set_ylabel("y",fontsize=16)
        ax[i].set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
        ax[i].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
        ax[i].set_title(u"在t="+Time[i]+u"时的误差分布",fontsize=16)
        cb = fig.colorbar(axcolor,ax=ax[i])
        cb.ax.tick_params(labelsize='large')


# %%
errorplot(model=model)


# %%
# 定义L-BFGS优化函数
L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, loss_ic=loss_ic, x=x, y=y, t=t,
                                    x_train_bc=x_train_bc, y_train_bc=y_train_bc, t_train_bc=t_train_bc, bc_train=bc_train,
                                    x_train_ic=x_train_ic, y_train_ic=y_train_ic, t_train_ic=t_train_ic, ic_train=ic_train,
                                    alpha=alpha, beta=beta, D=D)

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
    loss_i_list.append(L_BFGS_Optimizer.history[i].numpy()[3])


# %%
# 计算Adam L2范数误差
approximate = model(np.hstack((x_test,y_test,t_test))).numpy()   
approximate = approximate.reshape((p,q,r))
error = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error)/np.linalg.norm(real_test)))
print("MAE: {}".format(np.max(np.abs(error))))
print("MSE: {}".format(np.mean(np.abs(error))))


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
# 计算Adam L2范数误差
approximate = model(np.hstack((x_test,y_test,t_test))).numpy()   
approximate = approximate.reshape((p,q,r))
error = approximate - real_test
print("L2-error norm: {}".format(np.linalg.norm(error)/np.linalg.norm(real_test)))
print("MAE: {}".format(np.max(np.abs(error))))
print("MSE: {}".format(np.mean(np.abs(error))))


# %%
# 绘制误差图样
def lossplot(loss,loss_e,loss_b,loss_i): 
    fig = plt.figure(figsize=(20, 19))
    # Loss
    ax1 = fig.add_subplot(221) 
    ax1.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss[0:EPOCHS], label="Adam: Loss",c='b',basey=10)
    ax1.semilogy(np.linspace(EPOCHS,len(loss),len(loss)-EPOCHS),loss[EPOCHS:len(loss)], label="L-BFGS: Loss",c='r',basey=10)
    ax1.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax1.legend()
    # loss_e
    ax2 = fig.add_subplot(222) 
    ax2.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: Residual Loss",c='b',basey=10)
    ax2.semilogy(np.linspace(EPOCHS,len(loss_e),len(loss_e)-EPOCHS),loss_e[EPOCHS:len(loss_e)], label="L-BFGS: Residual Loss",c='r',basey=10)
    ax2.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax2.legend()
    # loss_b
    ax3 = fig.add_subplot(223) 
    ax3.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: Boundary Loss",c='b',basey=10)
    ax3.semilogy(np.linspace(EPOCHS,len(loss_b),len(loss_b)-EPOCHS),loss_b[EPOCHS:len(loss_b)], label="L-BFGS: Boundary Loss",c='r',basey=10)
    ax3.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax3.legend()
    # loss_i
    ax4 = fig.add_subplot(224) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_i[0:EPOCHS], label="Adam: Initial Loss",c='b',basey=10)
    ax4.semilogy(np.linspace(EPOCHS,len(loss_i),len(loss_i)-EPOCHS),loss_i[EPOCHS:len(loss_i)], label="L-BFGS: Initial Loss",c='r',basey=10)
    ax4.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax4.legend()


# %%
lossplot(loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,loss_i=loss_i_list)


# %%
errorplot(model=model)


# %%



