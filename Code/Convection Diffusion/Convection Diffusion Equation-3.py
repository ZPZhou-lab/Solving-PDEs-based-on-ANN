# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 对流扩散方程

# %%
# 实验环境
# python 3.9.3
# Tensorflow 2.4.1
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow_probability as tfp


# %%
# 显示中文
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Microsoft YaHei'],
}
rcParams.update(config)
plt.rcParams['axes.unicode_minus']=False


# %%
approximate = np.load('approximate_BFGS.npy')


# %%
for i in range(101):
    fig = plt.figure(figsize=(8,6),dpi=100)
    plt.imshow(approximate[:,:,i],vmin=0,vmax=1,cmap=plt.cm.hot,origin='lower')
    plt.xlabel('$x$',fontsize=20)
    plt.ylabel('$y$',fontsize=20)
    plt.title('温度在'+'$t=$'+format('%.3f' %(i*0.005))+'时的分布')
    cb = plt.colorbar()
    # cb.ax.tick_params(labelsize='large')
    plt.savefig(fname='approximate/approximate_'+str(i))

# %% [markdown]
# ## 梯形平板传热
# ### 方程类型：$\phi_t = (\phi_{xx}+\phi_{yy})$
# ### 其中，$x,y\in[0,1]$, $t\in[0,+\infty)$ 
# ### 边界条件：
# $\phi(0,y,t)=\phi(1,y,t)=\phi(x,0,t)=0$, $\phi(x,1,t)=1$, $\phi(x,y,0)=0$

# %%
# 训练点生成
# 内部训练点生成
n_residual = 800
xy_train = np.random.rand(n_residual,2)
xy_train = xy_train[(xy_train[:,1]<=0.25*xy_train[:,0]+0.75) & (xy_train[:,1]>=-0.25*xy_train[:,0]+0.25)]
# 边界点生成
n_bc = 31
up_train = np.transpose(np.vstack((np.linspace(0,1,n_bc),0.25*np.linspace(0,1,n_bc)+0.75)))
left_train = np.transpose(np.vstack((np.zeros(n_bc), np.linspace(0.25,0.75,n_bc))))
right_train = np.transpose(np.vstack((np.ones(n_bc), np.linspace(0,1,n_bc))))
down_train = np.transpose(np.vstack((np.linspace(0,1,n_bc),-0.25*np.linspace(0,1,n_bc)+0.25)))


# %%
# 训练点集绘图
fig = plt.figure(figsize=(8,8))
plt.scatter(xy_train[:,0],xy_train[:,1],s=12,label=r'区域内部随机采样')
plt.scatter(up_train[:,0],up_train[:,1],c='r',s=12,label=r'边界等距采样')
plt.scatter(left_train[:,0],left_train[:,1],c='r',s=12)
plt.scatter(right_train[:,0],right_train[:,1],c='r',s=12)
plt.scatter(down_train[:,0],down_train[:,1],c='r',s=12)
plt.xlabel('$x$',fontsize=16)
plt.ylabel('$y$',fontsize=16)
plt.legend()


# %%
# 训练点集与时间变量拼接
XY_train = np.vstack((xy_train,up_train,left_train,right_train,down_train))
t_use = 51
t = np.linspace(0,0.5,51)
xyt_train = np.zeros((XY_train.shape[0]*t_use,3),'float64')
for i in range(t_use):
    xyt_train[i*XY_train.shape[0]:(i+1)*XY_train.shape[0],0:2] = XY_train
    xyt_train[i*XY_train.shape[0]:(i+1)*XY_train.shape[0],2] = t[i]
# 取出初始时刻
t0_train = xyt_train[(xyt_train[:,2]==0) & (xyt_train[:,1]!=1)]
# 取出边界部分
u_train = xyt_train[(xyt_train[:,1]==1)]
l_train = xyt_train[(xyt_train[:,1]==(1-4*xyt_train[:,0]))]
r_train = xyt_train[(xyt_train[:,1]==(4*xyt_train[:,0]-3))]
d_train = xyt_train[(xyt_train[:,1]==0)]
xyt_train_bc = np.vstack((u_train,l_train,r_train,d_train))


# %%
# 导数残差训练点
x_train = np.reshape(xyt_train[:,0],[xyt_train.shape[0],1])
y_train = np.reshape(xyt_train[:,1],[xyt_train.shape[0],1])
t_train = np.reshape(xyt_train[:,2],[xyt_train.shape[0],1])
# 边界训练点
xyt_train_bc = np.vstack((u_train,l_train,r_train,d_train))
x_train_bc = xyt_train_bc[:,0].reshape(xyt_train_bc.shape[0],1)
y_train_bc = xyt_train_bc[:,1].reshape(xyt_train_bc.shape[0],1)
t_train_bc = xyt_train_bc[:,2].reshape(xyt_train_bc.shape[0],1)
# 初值训练点
x_train_ic = t0_train[:,0].reshape(t0_train.shape[0],1)
y_train_ic = t0_train[:,1].reshape(t0_train.shape[0],1)
t_train_ic = t0_train[:,2].reshape(t0_train.shape[0],1)


# %%
# 边界部分训练集真值
bc_train = np.hstack((np.ones(u_train.shape[0]),np.zeros(l_train.shape[0]),np.zeros(r_train.shape[0]),np.zeros(d_train.shape[0])))
bc_train = np.reshape(bc_train,[bc_train.shape[0],1])
# 初值部分训练集真值
ic_train = np.zeros(t0_train.shape[0])
ic_train = np.reshape(ic_train,[ic_train.shape[0],1])


# %%
# 生成测试格点
p = 101
q = 101
r = 101
X_test = np.linspace(0,1,p,endpoint=True) # 生成[0,1]区间p个点
Y_test = np.linspace(0,1,q,endpoint=True) # 生成[0,1]区间q个点
T_test = np.linspace(0,0.5,r,endpoint=True) # 生成[0,0.5]区间r个点
# 生成测试格点(x_i,y_j,t_k)
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
# 声明训练集为tf.Variable变量，作为网络输入
x = tf.Variable(tf.reshape(tf.cast(xyt_train[:,0], dtype=tf.float64),[xyt_train.shape[0],1]))
y = tf.Variable(tf.reshape(tf.cast(xyt_train[:,1], dtype=tf.float64),[xyt_train.shape[0],1]))
t = tf.Variable(tf.reshape(tf.cast(xyt_train[:,2], dtype=tf.float64),[xyt_train.shape[0],1]))


# %%
# 定义网络结构
struct = [3,50,50,50,50,1]
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
def residual(Psi_xx,Psi_yy,Psi_t):
    loss_e = tf.reduce_mean((Psi_t-(Psi_xx+Psi_yy))**2)
    return loss_e

# 定义边界损失函数
def loss_bc(model,x,y,t,bc_train):
    loss_b = tf.reduce_mean((model(tf.concat([x,y,t],1)) - bc_train)**2)
    return loss_b

# 定义初值损失函数
def loss_ic(model,x,y,t,ic_train):
    loss_i = tf.reduce_mean((model(tf.concat([x,y,t],1)) - ic_train)**2)
    return loss_i

# 边界损失动态权重调整函数
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

# 初值损失动态权重调整函
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
# 定义损失函数
def loss_function(model,residual,loss_bc,loss_ic,x,y,t,x_train_bc,y_train_bc,t_train_bc,x_train_ic,y_train_ic,t_train_ic,bc_train,ic_train):
    def loss(y_pred=None,y_true=None):    
        # 声明全局变量
        global alpha,beta
        global alpha_list,beta_list
        global method
        global epoch
        global loss_list,loss_e_list,loss_b_list,loss_i_list
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape3:
                    Psi = model(tf.concat([x,y,t],1))
                Psi_x, Psi_y, Psi_t = tape3.gradient(Psi,[x,y,t])
            Psi_xx = tape2.gradient(Psi_x,x)
            Psi_yy = tape2.gradient(Psi_y,y)
            loss_e = residual(Psi_xx=Psi_xx,Psi_yy=Psi_yy,Psi_t=Psi_t)
            loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,t=t_train_bc,bc_train=bc_train)
            loss_i = loss_ic(model=model,x=x_train_ic,y=y_train_ic,t=t_train_ic,ic_train=ic_train)
            loss_value = loss_e + alpha*loss_b + beta*loss_i
        if ((epoch+1)%100 == 0) and (method != 0): # 每100次更新一次权重
            loss_e_gradients = tape1.gradient(loss_e,model.variables)
            loss_b_gradients = tape1.gradient(loss_b,model.variables)
            loss_i_gradients = tape1.gradient(loss_i,model.variables)
            alpha = alpha_estimate(a=alpha,loss_e_gradients=loss_e_gradients,loss_b_gradients=loss_b_gradients,method=method)
            beta = beta_estimate(b=beta,loss_e_gradients=loss_e_gradients,loss_i_gradients=loss_i_gradients,method=method)
            alpha_list.append(alpha)
            beta_list.append(beta)
        loss_list.append(loss_value)
        loss_e_list.append(loss_e)
        loss_b_list.append(loss_b)
        loss_i_list.append(loss_i)
        epoch += 1
        return loss_value
    return loss


# %%
# L-BFGS优化器生成函数
def function_factory(model, residual, loss_bc, loss_ic, x, y, t, 
                    x_train_bc, y_train_bc, t_train_bc, bc_train,
                    x_train_ic, y_train_ic, t_train_ic, ic_train, alpha, beta):
    """定义一个tfp.optimizer.lbfgs_minimize所需的生成函数
    Args:
        model [in]: 一个`tf.keras.Model`的结构或它的继承.
        loss [in]: 损失函数，或损失函数列表(对应于多个损失).
        train_x [in]: 训练集变量.
        train_y [in]: 训练集标签.
        **kargs [in]: 模型训练时所需的其他参数(例如动态权重)
    Returns:
        返回一个具有如下返回值的函数:
            loss_value, gradients = f(model_parameters).
    """
    
    # 计算model中可训练参数的形状
    # 这一部分不需要更改
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)
    
    # 随后将使用tf.dynamic_switch和tf.dynamic_partition来更改形状，下面是准备工作
    # 这一部分不需要更改
    count = 0
    idx = [] # 指标
    part = [] # 分块下标
    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
    part = tf.constant(part)
    
    # 更新参数函数不需要进行更改
    @tf.function
    def assign_new_model_parameters(params_1d):
        """一个使用一维的tf.Tensor来更新模型参数的函数
        Args:
            params_1d [in]: 一个一维tf.Tensor，它代表模型的可训练参数.
        """
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))
    
    # 返回具有损失和梯度信息的函数
    # 这个部分需要根据具体模型进行更改
    @tf.function
    def f(params_1d):
        """一个可以被tfp.optimizer.lbfgs_minimize调用的函数.
        该函数由function_factory生成.
        Args:
            params_1d [in]: 一个一维tf.Tensor.
        Returns:
            对应于`params_1d`的损失及导数信息.
        """
        # 使用tf.GradientTape计算损失和导数
        # 在计算过程前更新模型的参数
        assign_new_model_parameters(params_1d)
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape3:
                    Psi = model(tf.concat([x,y,t],1))
                Psi_x, Psi_y, Psi_t = tape3.gradient(Psi,[x,y,t])
            Psi_xx = tape2.gradient(Psi_x,x)
            Psi_yy = tape2.gradient(Psi_y,y)
            loss_e = residual(Psi_xx=Psi_xx,Psi_yy=Psi_yy,Psi_t=Psi_t)
            loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,t=t_train_bc,bc_train=bc_train)
            loss_i = loss_ic(model=model,x=x_train_ic,y=y_train_ic,t=t_train_ic,ic_train=ic_train)
            # 计算损失
            loss_value = loss_e + alpha*loss_b + beta*loss_i
        # 计算梯度信息，并将其转换为一维tf.Tensor
        grads = tape1.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        # 迭代次数，并打印信息
        f.iter.assign_add(1)
        if f.iter%5000 == 0:
            tf.print("Iter:", f.iter, "loss:", loss_value)
        # 记录损失信息，便于之后取出
        tf.py_function(f.history.append, inp=[[loss_value,loss_e,loss_b,loss_i]], Tout=[])
        return loss_value, grads
    
    # 将相关信息存储为函数的成员，便于在函数之外调用它们
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []
    return f


# %%
# 编译模型，定义Adam优化器
model.compile(optimizer=Adam(learning_rate=0.001),
                loss=loss_function(model=model,residual=residual,loss_bc=loss_bc,loss_ic=loss_ic,x=x,y=y,t=t,
                                    x_train_bc=x_train_bc,y_train_bc=y_train_bc,t_train_bc=t_train_bc,
                                    x_train_ic=x_train_ic,y_train_ic=y_train_ic,t_train_ic=t_train_ic,
                                    bc_train=bc_train,ic_train=ic_train))
tf.config.run_functions_eagerly(True)


# %%
epoch = 0 # 记录训练次数
EPOCHS = 30000 # 定义总的训练次数
alpha = 100 # 边界权重初始化
beta = 100 # 初值权重初始化
method = 1 # 启用第一种动态权重更新策略
# 存储损失信息
loss_list = []
loss_e_list = []
loss_b_list = []
loss_i_list = []
# 存储动态权重信息
alpha_list = [alpha]
beta_list = [beta]


# %%
# 训练Adam优化器模型
model.fit(x=tf.concat([x,y,t],1),y=tf.zeros_like(x),batch_size=50000,epochs=EPOCHS,verbose=0)


# %%
# 定义L-BFGS优化函数
L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, loss_ic=loss_ic, x=x, y=y, t=t,
                                    x_train_bc=x_train_bc, y_train_bc=y_train_bc, t_train_bc=t_train_bc, bc_train=bc_train,
                                    x_train_ic=x_train_ic, y_train_ic=y_train_ic, t_train_ic=t_train_ic, ic_train=ic_train,
                                    alpha=alpha, beta=beta)

# 初始化L-BFGS优化器参数
init_params = tf.dynamic_stitch(L_BFGS_Optimizer.idx, model.trainable_variables)

# 使用L-BFGS优化器训练模型
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=L_BFGS_Optimizer, 
                                        initial_position=init_params, tolerance=1e-10, max_iterations=30000)

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
# 梯形数据处理函数
def data_process(data):
    x_check = np.linspace(0,1,data.shape[0])
    y_check = np.linspace(0,1,data.shape[1])
    for k in range(data.shape[2]):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (y_check[j] < 1 - 4*x_check[i]) or (y_check[j] < 4*x_check[i] -3):
                    data[i,j,k] = np.nan
    return data


# %%
# 计算逼近值
approximate = model(np.hstack((x_test,y_test,t_test))).numpy()   
approximate = approximate.reshape((p,q,r))
# 更改为梯形
approximate = data_process(data=approximate)


# %%
def imageplot(approximate,step):
    vmin = 0
    vmax = 1
    Time = [format(0.005*step*i, '.3f') for i in range(9)]
    X = np.linspace(0,1,101)
    Y = np.linspace(0,1,101)
    fig, ax = plt.subplots(3,3, figsize=(16,14))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.flatten()
    for i in range(9):b
        axcolor = ax[i].imshow(approximate[:,:,i*step+1],vmin=vmin,vmax=vmax,cmap=plt.cm.hot,origin='lower')
        ax[i].set_xlabel("$x$",fontsize=16)
        ax[i].set_ylabel("$y$",fontsize=16)
        ax[i].set_title(r"在$\mathrm{t=}$"+Time[i]+r"时的温度分布",fontsize=16)
        cb = fig.colorbar(axcolor,ax=ax[i])
        cb.ax.tick_params(labelsize='large')


# %%
imageplot(approximate,1)


# %%
# 绘制误差图样
def lossplot(loss,loss_e,loss_b,loss_i,alpha,beta): 
    fig = plt.figure(figsize=(20, 20))
    # Loss
    ax1 = fig.add_subplot(321) 
    ax1.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss[0:EPOCHS], label="Adam: Loss",c='b',basey=10)
    ax1.semilogy(np.linspace(EPOCHS,len(loss),len(loss)-EPOCHS),loss[EPOCHS:len(loss)], label="L-BFGS: Loss",c='r',basey=10)
    ax1.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax1.legend()
    # loss_e
    ax2 = fig.add_subplot(322) 
    ax2.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: Residual Loss",c='b',basey=10)
    ax2.semilogy(np.linspace(EPOCHS,len(loss_e),len(loss_e)-EPOCHS),loss_e[EPOCHS:len(loss_e)], label="L-BFGS: Residual Loss",c='r',basey=10)
    ax2.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax2.legend()
    # loss_b
    ax3 = fig.add_subplot(323) 
    ax3.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: Boundary Loss",c='b',basey=10)
    ax3.semilogy(np.linspace(EPOCHS,len(loss_b),len(loss_b)-EPOCHS),loss_b[EPOCHS:len(loss_b)], label="L-BFGS: Boundary Loss",c='r',basey=10)
    ax3.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax3.legend()
    # loss_i
    ax4 = fig.add_subplot(324) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_i[0:EPOCHS], label="Adam: Initial Loss",c='b',basey=10)
    ax4.semilogy(np.linspace(EPOCHS,len(loss_i),len(loss_i)-EPOCHS),loss_i[EPOCHS:len(loss_i)], label="L-BFGS: Initial Loss",c='r',basey=10)
    ax4.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax4.legend()
    # alpha,beta
    ax5 = fig.add_subplot(325)
    ax5.plot(np.linspace(1,len(alpha),len(alpha)),alpha,label="Alpha",c='b')
    ax5.plot(np.linspace(1,len(beta),len(beta)),beta,label="Beta",c='r')
    ax5.legend()


# %%
lossplot(loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,loss_i=loss_i_list)


# %%



