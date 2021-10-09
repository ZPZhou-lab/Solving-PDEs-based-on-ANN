# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

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
# ### 方程：
# ## $\frac{\partial \vec u}{\partial t}+(\vec u\cdot \nabla)\vec u = -\nabla p + \frac{1}{Re}\nabla^2 \vec u$
# ### 其中：$\vec u = \vec u_{\Gamma}, $ on   $ \Gamma_D,$  $\nabla \cdot \vec u=0, $ in   $\Omega$

# %%
fig = plt.figure(figsize=(6,8))
plt.scatter(xy_train_RAR[0:1000,0],xy_train_RAR[0:1000,1],color='blue',s=2,alpha=0.2,label='Initial Training Points')
plt.scatter(RAR[:,0],RAR[:,1],color='red',s=5,label='RAR Training Points')
plt.legend()


# %%
xy_train_RAR = np.load('xy_train_RAR.npy')


# %%
RAR = xy_train_RAR[1000:,:]


# %%
import numpy as np
import scipy.io as io
mat_path = 'xy_train_RAR.mat'
mat = xy_train_RAR
io.savemat(mat_path, {'name': mat})


# %%



# %%



# %%
# 生成内部训练点
n = 2500
xy_train = np.random.random([n,2])
xy_train[:,0] = xy_train[:,0]*1.5 - 0.5
xy_train[:,1] = xy_train[:,1]*2 - 0.5
# 绘制内部训练节点分布
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax1.scatter(xy_train[:,0],xy_train[:,1],s=5)

# 生成边界训练节点
n_bc = 101
x0_train = np.hstack((np.zeros([n_bc,1])-0.5,np.reshape(np.linspace(-0.5,1.5,n_bc),[n_bc,1])))
x1_train = np.hstack((np.ones([n_bc,1]),np.reshape(np.linspace(-0.5,1.5,n_bc),[n_bc,1])))
y0_train = np.hstack((np.reshape(np.linspace(-0.5,1,n_bc),[n_bc,1]),np.zeros([n_bc,1]) - 0.5))
y1_train = np.hstack((np.reshape(np.linspace(-0.5,1,n_bc),[n_bc,1]),np.ones([n_bc,1]) + 0.5))
xy_train_bc = np.vstack((x0_train,x1_train,y0_train,y1_train))
x_train_bc = xy_train_bc[:,0].reshape(xy_train_bc.shape[0],1)
y_train_bc = xy_train_bc[:,1].reshape(xy_train_bc.shape[0],1)
# 绘制边界训练节点分布
ax2 = fig.add_subplot(132)
ax2.scatter(xy_train_bc[:,0],xy_train_bc[:,1],c=[[1,0.4,0.4]],s=5)

# 将边界添加到内部训练点
ax3 = fig.add_subplot(133)
ax3.scatter(xy_train[:,0],xy_train[:,1],s=5)
ax3.scatter(xy_train_bc[:,0],xy_train_bc[:,1],c=[[1,0.4,0.4]],s=5)
# xy_train = np.vstack((xy_train,xy_train_bc))


# %%
# 生成测试格点
p = 101
q = 101
X_test = np.linspace(-0.5,1,p,endpoint=True) # 生成[0,1]区间p个点
Y_test = np.linspace(-0.5,1.5,q,endpoint=True) # 生成[0,1]区间q个点
# 生成测试格点(x_j,y_j)
xy_test = np.zeros((p*q,2),"float64")
for i in range(p):
    for j in range(q):
        xy_test[(i*q+j),0]=X_test[i]
        xy_test[(i*q+j),1]=Y_test[j]     
x_test = np.reshape(xy_test[:,0],[p*q,1])
y_test = np.reshape(xy_test[:,1],[p*q,1])


# %%
v_0 = 1/40
zeta = 1/(2*v_0) - np.sqrt(1/(4*v_0**2)+4*np.pi**2)
# 测试集真值
u_real = 1 - np.exp(zeta*x_test)*np.cos(2*np.pi*y_test)
v_real = zeta/(2*np.pi)*np.exp(zeta*x_test)*np.sin(2*np.pi*y_test)
p_real = 0.5*(1 - np.exp(2*zeta*x_test))
# 变换形状
u_real = np.reshape(u_real,[p,q])
v_real = np.reshape(v_real,[p,q])
p_real = np.reshape(p_real,[p,q])
# 计算边界训练点真值
bc_train_u = 1 - np.exp(zeta*x_train_bc)*np.cos(2*np.pi*y_train_bc)
bc_train_u = np.reshape(bc_train_u,[bc_train_u.shape[0],1])
bc_train_v = zeta/(2*np.pi)*np.exp(zeta*x_train_bc)*np.sin(2*np.pi*y_train_bc)
bc_train_v = np.reshape(bc_train_v,[bc_train_v.shape[0],1])


# %%
# 训练集准备
x = tf.Variable(tf.reshape(tf.cast(xy_train[:,0], dtype=tf.float64),[xy_train.shape[0],1]))
y = tf.Variable(tf.reshape(tf.cast(xy_train[:,1], dtype=tf.float64),[xy_train.shape[0],1]))


# %%
# 定义残差函数(导数增长损失函数)
def residual(Psi_u,Psi_u_x,Psi_u_xx,Psi_u_y,Psi_u_yy,Psi_v,Psi_v_x,Psi_v_xx,Psi_v_y,Psi_v_yy,Psi_p_x,Psi_p_y,v_0=1/40):
    loss_d_1 = Psi_u*Psi_u_x + Psi_v*Psi_u_y + Psi_p_x - v_0*(Psi_u_xx + Psi_u_yy)
    loss_d_2 = Psi_u*Psi_v_x + Psi_v*Psi_v_y + Psi_p_y - v_0*(Psi_v_xx + Psi_v_yy)
    loss_d_3 = Psi_u_x + Psi_v_y
    loss_e = tf.reduce_mean(loss_d_1**2 + loss_d_2**2 + loss_d_3**2)
    return loss_e

# 定义边界损失函数
def loss_bc(model,x,y,bc_train_u,bc_train_v):
    l_bc = bc_train_u.shape[0]
    loss_b = tf.reduce_mean((tf.reshape(model(tf.concat([x,y],1))[:,0],(l_bc,1)) - bc_train_u)**2 + (tf.reshape(model(tf.concat([x,y],1))[:,1],(l_bc,1)) - bc_train_v)**2)
    return loss_b

# 动态权重
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

# L-BFGS生成函数
def function_factory(model, residual, loss_bc, x, y, x_train_bc, y_train_bc, bc_train_u, bc_train_v, alpha):
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
    # 计算维度
    n_dim = model(tf.concat([x,y],1)).shape[0]
    
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
                    Psi_u = tf.reshape(model(tf.concat([x,y],1))[:,0],[n_dim,1])
                    Psi_v = tf.reshape(model(tf.concat([x,y],1))[:,1],[n_dim,1])
                    Psi_p = tf.reshape(model(tf.concat([x,y],1))[:,2],[n_dim,1])
                Psi_u_x, Psi_u_y = tape3.gradient(Psi_u,[x,y])
                Psi_v_x, Psi_v_y = tape3.gradient(Psi_v,[x,y])
                Psi_p_x, Psi_p_y = tape3.gradient(Psi_p,[x,y])
            Psi_u_xx = tape2.gradient(Psi_u_x,x)
            Psi_u_yy = tape2.gradient(Psi_u_y,y)
            Psi_v_xx = tape2.gradient(Psi_v_x,x)
            Psi_v_yy = tape2.gradient(Psi_v_y,y)
            loss_e = residual(Psi_u=Psi_u,Psi_u_x=Psi_u_x,Psi_u_xx=Psi_u_xx,Psi_u_y=Psi_u_y,Psi_u_yy=Psi_u_yy,
                              Psi_v=Psi_v,Psi_v_x=Psi_v_x,Psi_v_xx=Psi_v_xx,Psi_v_y=Psi_v_y,Psi_v_yy=Psi_v_yy,
                              Psi_p_x=Psi_p_x,Psi_p_y=Psi_p_y,v_0=1/40)
            loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,bc_train_u=bc_train_u,bc_train_v=bc_train_v)
            # calculate the loss
            loss_value = loss_e + alpha*loss_b
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
def loss_function(model,x,y,residual,loss_bc,alpha_estimate,method):
    def loss(y_true=None,y_pred=None):
        global alpha_index
        global alpha_list
        global n_dim
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape3:            
                    Psi_u = tf.reshape(model(tf.concat([x,y],1))[:,0],[n_dim,1])
                    Psi_v = tf.reshape(model(tf.concat([x,y],1))[:,1],[n_dim,1])
                    Psi_p = tf.reshape(model(tf.concat([x,y],1))[:,2],[n_dim,1])
                Psi_u_x, Psi_u_y = tape3.gradient(Psi_u,[x,y])
                Psi_v_x, Psi_v_y = tape3.gradient(Psi_v,[x,y])
                Psi_p_x, Psi_p_y = tape3.gradient(Psi_p,[x,y])
            Psi_u_xx = tape2.gradient(Psi_u_x,x)
            Psi_u_yy = tape2.gradient(Psi_u_y,y)
            Psi_v_xx = tape2.gradient(Psi_v_x,x)
            Psi_v_yy = tape2.gradient(Psi_v_y,y)
            loss_e = residual(Psi_u=Psi_u,Psi_u_x=Psi_u_x,Psi_u_xx=Psi_u_xx,Psi_u_y=Psi_u_y,Psi_u_yy=Psi_u_yy,
                              Psi_v=Psi_v,Psi_v_x=Psi_v_x,Psi_v_xx=Psi_v_xx,Psi_v_y=Psi_v_y,Psi_v_yy=Psi_v_yy,
                              Psi_p_x=Psi_p_x,Psi_p_y=Psi_p_y,v_0=1/40)
            loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,bc_train_u=bc_train_u,bc_train_v=bc_train_v)
            loss_all = loss_e + alpha_list[len(alpha_list)-1]*loss_b
        if ((alpha_index+1)%100 == 0) and (method != 0): # 每100次更新一次权重alpha
            loss_e_gradients = tape1.gradient(loss_e,model.variables)
            loss_b_gradients = tape1.gradient(loss_b,model.variables)
            alpha = alpha_estimate(a=alpha_list[len(alpha_list)-1],loss_e_gradients=loss_e_gradients,loss_b_gradients=loss_b_gradients,method=method)
            alpha_list.append(alpha)
        alpha_index += 1
        return loss_all
    return loss


# %%
# 定义网络结构
struct = [2,30,30,30,3]
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
alpha_list = [100]
alpha_index = 0
n_dim = model(tf.concat([x,y],1)).shape[0]


# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_function(model=model,x=x,y=y,residual=residual,loss_bc=loss_bc,alpha_estimate=alpha_estimate,method=1))
tf.config.run_functions_eagerly(True)


# %%
model.fit(x=tf.concat([x,y],1),y=tf.zeros_like(x),batch_size=30000,epochs=200,verbose=1)


# %%
EPOCHS = 30000 # 训练次数
alpha = 100 # 边界权重初始化
method = 1
loss_list = []
loss_e_list = []
loss_b_list = []
alpha_list = [alpha]
n_dim = model(tf.concat([x,y],1)).shape[0]
for epoch in range(EPOCHS):
    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape3:            
                Psi_u = tf.reshape(model(tf.concat([x,y],1))[:,0],[n_dim,1])
                Psi_v = tf.reshape(model(tf.concat([x,y],1))[:,1],[n_dim,1])
                Psi_p = tf.reshape(model(tf.concat([x,y],1))[:,2],[n_dim,1])
            Psi_u_x, Psi_u_y = tape3.gradient(Psi_u,[x,y])
            Psi_v_x, Psi_v_y = tape3.gradient(Psi_v,[x,y])
            Psi_p_x, Psi_p_y = tape3.gradient(Psi_p,[x,y])
        Psi_u_xx = tape2.gradient(Psi_u_x,x)
        Psi_u_yy = tape2.gradient(Psi_u_y,y)
        Psi_v_xx = tape2.gradient(Psi_v_x,x)
        Psi_v_yy = tape2.gradient(Psi_v_y,y)
        loss_e = residual(Psi_u=Psi_u,Psi_u_x=Psi_u_x,Psi_u_xx=Psi_u_xx,Psi_u_y=Psi_u_y,Psi_u_yy=Psi_u_yy,
                          Psi_v=Psi_v,Psi_v_x=Psi_v_x,Psi_v_xx=Psi_v_xx,Psi_v_y=Psi_v_y,Psi_v_yy=Psi_v_yy,
                          Psi_p_x=Psi_p_x,Psi_p_y=Psi_p_y,v_0=1/40)
        loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,bc_train_u=bc_train_u,bc_train_v=bc_train_v)
        loss = loss_e + alpha*loss_b
    if ((epoch+1)%100 == 0) and (method != 0): # 每100次更新一次权重alpha
        loss_e_gradients = tape1.gradient(loss_e,model.variables)
        loss_b_gradients = tape1.gradient(loss_b,model.variables)
        alpha = alpha_estimate(a=alpha,loss_e_gradients=loss_e_gradients,loss_b_gradients=loss_b_gradients,method=method)
        alpha_list.append(alpha)
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
approximate_u = model(np.hstack((x_test,y_test)))[:,0].numpy().reshape((p,q))
approximate_v = model(np.hstack((x_test,y_test)))[:,1].numpy().reshape((p,q))
approximate_p = model(np.hstack((x_test,y_test)))[:,2].numpy().reshape((p,q))
# 压强均值平移
approximate_p = approximate_p - (np.mean(approximate_p) - np.mean(p_real))
error_u = approximate_u - u_real
error_v = approximate_v - v_real
error_p = approximate_p - p_real
print("L2-error norm of u: {}".format(np.linalg.norm(error_u)/np.linalg.norm(u_real)))
print("L2-error norm of v: {}".format(np.linalg.norm(error_v)/np.linalg.norm(v_real)))
print("L2-error norm of p: {}".format(np.linalg.norm(error_p)/np.linalg.norm(p_real)))


# %%
# 定义L-BFGS优化函数
L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, x=x, y=y, 
                                    x_train_bc=x_train_bc, y_train_bc=y_train_bc, 
                                    bc_train_u=bc_train_u, bc_train_v=bc_train_v, alpha=alpha_list[len(alpha_list)-1])

# 初始化L-BFGS优化器参数
init_params = tf.dynamic_stitch(L_BFGS_Optimizer.idx, model.trainable_variables)

# 使用L-BFGS优化器训练模型
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=L_BFGS_Optimizer, initial_position=init_params, tolerance=1e-08, max_iterations=10000)

# 最后一次训练后，将参数更新到模型
L_BFGS_Optimizer.assign_new_model_parameters(results.position)


# %%
# 取出L-BFGS训练的损失信息
for i in range(len(L_BFGS_Optimizer.history)):
    loss_list.append(L_BFGS_Optimizer.history[i].numpy()[0])
    loss_e_list.append(L_BFGS_Optimizer.history[i].numpy()[1])
    loss_b_list.append(L_BFGS_Optimizer.history[i].numpy()[2])


# %%
# 计算Adam L2范数误差
approximate_u = model(np.hstack((x_test,y_test)))[:,0].numpy().reshape((p,q))
approximate_v = model(np.hstack((x_test,y_test)))[:,1].numpy().reshape((p,q))
approximate_p = model(np.hstack((x_test,y_test)))[:,2].numpy().reshape((p,q))
error_u = approximate_u - u_real
error_v = approximate_v - v_real
error_p = approximate_p - p_real
print("L2-error norm of u: {}".format(np.linalg.norm(error_u)/np.linalg.norm(u_real)))
print("L2-error norm of v: {}".format(np.linalg.norm(error_v)/np.linalg.norm(v_real)))
print("L2-error norm of p: {}".format(np.linalg.norm(error_p)/np.linalg.norm(p_real)))


# %%
# 残差点的绝对误差
def absolute_residual(Psi_u,Psi_u_x,Psi_u_xx,Psi_u_y,Psi_u_yy,
                     Psi_v,Psi_v_x,Psi_v_xx,Psi_v_y,Psi_v_yy,Psi_p_x,Psi_p_y,v_0=1/40):
    loss_1 = tf.abs(Psi_u*Psi_u_x + Psi_v*Psi_u_y + Psi_p_x - v_0*(Psi_u_xx + Psi_u_yy))
    loss_2 = tf.abs(Psi_u*Psi_v_x + Psi_v*Psi_v_y + Psi_p_y - v_0*(Psi_v_xx + Psi_v_yy))
    loss_3 = tf.abs(Psi_u_x + Psi_v_y)
    return (loss_1 + loss_2 + loss_3)

# 新增训练点生成函数
def MonteCarlo_residual(model,num,m,epsilon):
    # 随机采样
    xy_train_temp = np.random.random([num,2])
    x_temp = tf.Variable(tf.reshape(tf.cast(xy_train_temp[:,0], dtype=tf.float64),[xy_train_temp.shape[0],1]))
    y_temp = tf.Variable(tf.reshape(tf.cast(xy_train_temp[:,1], dtype=tf.float64),[xy_train_temp.shape[0],1]))
    n_dim = model(tf.concat([x_temp,y_temp],1)).shape[0]
    # 计算残差点的绝对误差均值
    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            Psi_u = tf.reshape(model(tf.concat([x_temp,y_temp],1))[:,0],[n_dim,1])
            Psi_v = tf.reshape(model(tf.concat([x_temp,y_temp],1))[:,1],[n_dim,1])
            Psi_p = tf.reshape(model(tf.concat([x_temp,y_temp],1))[:,2],[n_dim,1])
        Psi_u_x, Psi_u_y = tape2.gradient(Psi_u,[x_temp,y_temp])
        Psi_v_x, Psi_v_y = tape2.gradient(Psi_v,[x_temp,y_temp])
        Psi_p_x, Psi_p_y = tape2.gradient(Psi_p,[x_temp,y_temp])
    Psi_u_xx = tape1.gradient(Psi_u_x,x_temp)
    Psi_u_yy = tape1.gradient(Psi_u_y,y_temp)
    Psi_v_xx = tape1.gradient(Psi_v_x,x_temp)
    Psi_v_yy = tape1.gradient(Psi_v_y,y_temp)
    L = absolute_residual(Psi_u=Psi_u,Psi_u_x=Psi_u_x,Psi_u_xx=Psi_u_xx,Psi_u_y=Psi_u_y,Psi_u_yy=Psi_u_yy,
                         Psi_v=Psi_v,Psi_v_x=Psi_v_x,Psi_v_xx=Psi_v_xx,Psi_v_y=Psi_v_y,Psi_v_yy=Psi_v_yy,
                         Psi_p_x=Psi_p_x,Psi_p_y=Psi_p_y).numpy()
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
        L_BFGS_Optimizer = function_factory(model=model, residual=residual, loss_bc=loss_bc, x=x, y=y, 
                                            x_train_bc=x_train_bc, y_train_bc=y_train_bc, 
                                            bc_train_u=bc_train_u, bc_train_v=bc_train_v, alpha=alpha)
        init_params = tf.dynamic_stitch(L_BFGS_Optimizer.idx, model.trainable_variables)
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=L_BFGS_Optimizer, 
                                               initial_position=init_params, tolerance=1e-08, max_iterations=10000)
        L_BFGS_Optimizer.assign_new_model_parameters(results.position)
        add_point,mean_L = MonteCarlo_residual(model,num,m,epsilon)
    
    # 返回模型，训练节点和最新残差
    return model,xy_train,mean_L


# %%
modelRAR,xy_train_RAR,mean_L = RAR(model=model,xy_train=xy_train,num=10**5,m=1,epsilon=2*0.001)


# %%
modelRAR,xy_train_RAR,mean_L = RAR(model=model,xy_train=xy_train,num=10**5,m=1,epsilon=0.001)


# %%
# 计算Adam L2范数误差
approximate_u = modelRAR(np.hstack((x_test,y_test)))[:,0].numpy().reshape((p,q))
approximate_v = modelRAR(np.hstack((x_test,y_test)))[:,1].numpy().reshape((p,q))
approximate_p = modelRAR(np.hstack((x_test,y_test)))[:,2].numpy().reshape((p,q))
error_u = approximate_u - u_real
error_v = approximate_v - v_real
error_p = approximate_p - p_real
print("L2-error norm of u: {}".format(np.linalg.norm(error_u)/np.linalg.norm(u_real)))
print("L2-error norm of v: {}".format(np.linalg.norm(error_v)/np.linalg.norm(v_real)))
print("L2-error norm of p: {}".format(np.linalg.norm(error_p)/np.linalg.norm(p_real)))


# %%
# 绘制误差图样
def errorplot(model,loss,loss_e,loss_b,alpha,rate=0.9): 
    X,Y = np.meshgrid(Y_test,X_test)
    fig = plt.figure(figsize=(20, 36))
    l = p*q
    approximate_u = model(np.hstack((x_test,y_test)))[:,0].numpy().reshape((p,q))
    approximate_v = model(np.hstack((x_test,y_test)))[:,1].numpy().reshape((p,q))
    approximate_p = model(np.hstack((x_test,y_test)))[:,2].numpy().reshape((p,q))
    error_u = approximate_u - u_real
    error_v = approximate_v - v_real
    error_p = approximate_p - p_real
    # 逼近曲面u
    ax1 = fig.add_subplot(521, projection='3d')
    ax1.plot_surface(X,Y,approximate_u,cmap='viridis', edgecolor='none')
    plt.title('Approximate u')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    ax1.set_zlabel('$u(x,y)$')
    ax1.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 误差曲面u
    ax2 = fig.add_subplot(522, projection='3d')
    ax2.plot_surface(X,Y,error_u,cmap='viridis', edgecolor='none')
    plt.title('Error u')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    ax2.set_zlabel('Error')
    ax2.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 逼近曲面v
    ax1 = fig.add_subplot(523, projection='3d')
    ax1.plot_surface(X,Y,approximate_v,cmap='viridis', edgecolor='none')
    plt.title('Approximate v')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    ax1.set_zlabel('$v(x,y)$')
    ax1.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 误差曲面v
    ax2 = fig.add_subplot(524, projection='3d')
    ax2.plot_surface(X,Y,error_v,cmap='viridis', edgecolor='none')
    plt.title('Error v')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    ax2.set_zlabel('Error')
    ax2.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 逼近曲面p
    ax1 = fig.add_subplot(525, projection='3d')
    ax1.plot_surface(X,Y,approximate_p,cmap='viridis', edgecolor='none')
    plt.title('Approximate p')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    ax1.set_zlabel('$p(x,y)$')
    ax1.view_init(azim=-145)    # 方位角
    plt.grid(True)
    # 误差曲面p
    ax2 = fig.add_subplot(526, projection='3d')
    ax2.plot_surface(X,Y,error_p,cmap='viridis', edgecolor='none')
    plt.title('Error p')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    ax2.set_zlabel('Error')
    ax2.view_init(azim=-145)    # 方位角
    plt.grid(True)
    
    # Loss
    ax3 = fig.add_subplot(527) 
    ax3.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss[0:EPOCHS], label="Adam: Loss",c='b',basey=10)
    ax3.semilogy(np.linspace(EPOCHS,len(loss),len(loss)-EPOCHS),loss[EPOCHS:len(loss)], label="L-BFGS: Loss",c='r',basey=10)
    ax3.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax3.legend()
    # loss_e
    ax4 = fig.add_subplot(528) 
    ax4.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_e[0:EPOCHS], label="Adam: Residual Loss",c='b',basey=10)
    ax4.semilogy(np.linspace(EPOCHS,len(loss_e),len(loss_e)-EPOCHS),loss_e[EPOCHS:len(loss_e)], label="L-BFGS: Residual Loss",c='r',basey=10)
    ax4.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax4.legend()
    # loss_b
    ax5 = fig.add_subplot(529) 
    ax5.semilogy(np.linspace(1,EPOCHS,EPOCHS),loss_b[0:EPOCHS], label="Adam: Boundary Loss",c='b',basey=10)
    ax5.semilogy(np.linspace(EPOCHS,len(loss_b),len(loss_b)-EPOCHS),loss_b[EPOCHS:len(loss_b)], label="L-BFGS: Boundary Loss",c='r',basey=10)
    ax5.axvline(x=EPOCHS,ls="--",c='g',lw=2)
    ax5.legend()
    # alpha
    ax6 = fig.add_subplot(5,2,10) 
    ax6.plot(np.linspace(1,len(alpha),len(alpha)),alpha, label="Alpha",c='r')
    ax6.legend()


# %%
# 绘图查看
errorplot(model,loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,alpha=alpha_list,rate=0.8)


# %%
# 绘图查看
errorplot(modelRAR,loss=loss_list,loss_e=loss_e_list,loss_b=loss_b_list,alpha=alpha_list,rate=0.8)


# %%
# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.save('./save/NS-Q1/model.ckpt')


# %%
# 读取模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore('./save/NS-Q1/model.ckpt-1')


