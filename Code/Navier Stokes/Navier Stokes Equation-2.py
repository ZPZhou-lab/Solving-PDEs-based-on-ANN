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
import scipy.io


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
# 定义网络
struct = [3,50,50,50,50,50,50,50,50,3]
tf.keras.backend.set_floatx("float32")
model = keras.models.Sequential()
# 采用Xavier初始化方法
for layer in range(len(struct)-1):
    val = tf.cast(tf.sqrt(6/(struct[layer]+struct[layer+1])),dtype=tf.float32)
    if layer == len(struct)-2:
        model.add(keras.layers.Dense(units=struct[layer+1],kernel_initializer=initializers.random_uniform(-val,val)))
    elif layer == 0:
        model.add(keras.layers.Dense(units=struct[layer+1],input_dim=struct[layer],activation='tanh',
                                     kernel_initializer=initializers.random_uniform(-val,val)))
    else:
        model.add(keras.layers.Dense(units=struct[layer+1],activation='tanh',
                                     kernel_initializer=initializers.random_uniform(-val,val)))


# %%
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore('NS-Q2/save/model_step5Complete.ckpt-1')


# %%
alpha_list = np.load('NS-Q2/alpha_list.npy')
beta_list = np.load('NS-Q2/beta_list.npy')
loss_list = np.load('NS-Q2/loss_list.npy')
loss_e_list = np.load('NS-Q2/loss_e_list.npy')
loss_i_list = np.load('NS-Q2/loss_i_list.npy')
loss_b_list = np.load('NS-Q2/loss_b_list.npy')


# %%
loss_list.shape[0]


# %%
fig = plt.figure(figsize=(14,8))
step1 = np.linspace(1,5000,5000)
plt.semilogy(step1,loss_e_list[0:5000],base=10,c='b',linewidth=1)
plt.semilogy(step1,loss_b_list[0:5000],base=10,c='r',linewidth=1)
plt.semilogy(step1,loss_i_list[0:5000],base=10,c='orange',linewidth=1)
step2 = np.linspace(5001,10000,5000)
plt.semilogy(step2,loss_e_list[5000:10000],base=10,c='b',linewidth=1)
plt.semilogy(step2,loss_b_list[5000:10000],base=10,c='r',linewidth=1)
plt.semilogy(step2,loss_i_list[5000:10000],base=10,c='orange',linewidth=1)
step3 = np.linspace(10001,60000,50000)
plt.semilogy(step3,loss_e_list[10000:60000],base=10,c='b',linewidth=1)
plt.semilogy(step3,loss_b_list[10000:60000],base=10,c='r',linewidth=1)
plt.semilogy(step3,loss_i_list[10000:60000],base=10,c='orange',linewidth=1)
step4 = np.linspace(60001,110000,50000)
plt.semilogy(step4,loss_e_list[60000:110000],base=10,c='b',linewidth=1)
plt.semilogy(step4,loss_b_list[60000:110000],base=10,c='r',linewidth=1)
plt.semilogy(step4,loss_i_list[60000:110000],base=10,c='orange',linewidth=1)
step5 = np.linspace(110001,loss_list.shape[0],loss_list.shape[0]-110000)
plt.semilogy(step5,loss_e_list[110000:],base=10,c='b',linewidth=1)
plt.semilogy(step5,loss_b_list[110000:],base=10,c='r',linewidth=1)
plt.semilogy(step5,loss_i_list[110000:],base=10,c='orange',linewidth=1)


# %%
fig = plt.figure(figsize=(12,8))
step1 = np.linspace(1,50,50)
plt.plot(step1,alpha_list[0:50],c='b',linewidth=1,label=r'$\alpha$')
plt.plot(step1,beta_list[0:50],c='r',linewidth=1,label=r'$\beta$')
step2 = np.linspace(51,100,50)
plt.plot(step2,alpha_list[50:100],c='b',linewidth=1)
plt.plot(step2,beta_list[50:100],c='r',linewidth=1)
step3 = np.linspace(101,600,500)
plt.plot(step3,alpha_list[100:600],c='b',linewidth=1)
plt.plot(step3,beta_list[100:600],c='r',linewidth=1)
step4 = np.linspace(601,1100,500)
plt.plot(step4,alpha_list[600:1100],c='b',linewidth=1)
plt.plot(step4,beta_list[600:1100],c='r',linewidth=1)
plt.xlabel(r'更新次数')
plt.legend(loc='lower right')


# %%
X_star.shape


# %%
error_u_list = []
error_v_list = []
error_p_list = []
approximate_U = np.zeros((5000,51))
approximate_V = np.zeros((5000,51))
approximate_P = np.zeros((5000,51))
xyt_test = np.zeros((5000,3))
xyt_test[:,0:2] = X_star
for t in range(51):
    xyt_test[:,2] = t*0.1
    approximate_U[:,t] = model(xyt_test)[:,0]
    approximate_V[:,t] = model(xyt_test)[:,1]
    approximate_P[:,t] = model(xyt_test)[:,2]
    approximate_u = model(xyt_test)[:,0]
    approximate_v = model(xyt_test)[:,1]
    approximate_p = model(xyt_test)[:,2]
    approximate_p = approximate_p - (np.mean(approximate_p) - np.mean(P[:,t]))
    error_u = approximate_u - U[:,t]
    error_v = approximate_v - V[:,t]
    error_p = approximate_p - P[:,t]
    error_u_list.append(np.linalg.norm(error_u)/np.linalg.norm(U[:,t]))
    error_v_list.append(np.linalg.norm(error_v)/np.linalg.norm(V[:,t]))
    error_p_list.append(np.linalg.norm(error_p)/np.linalg.norm(P[:,t]))


# %%
fig = plt.figure(figsize=(10,8))
plt.plot(np.linspace(0,5,51),error_u_list,c='r',linewidth=1,label=r'$\varepsilon_u$(%)')
plt.plot(np.linspace(0,5,51),error_v_list,c='b',linewidth=1,label=r'$\varepsilon_v$(%)')
plt.plot(np.linspace(0,5,51),error_p_list,c='orange',linewidth=1,label=r'$\varepsilon_p$(%)')
plt.legend(fontsize=15)


# %%
import numpy as np
import scipy.io as io
mat_path = 'beta_list.mat'
io.savemat(mat_path, {'beta_list': beta_list})


# %%
data = scipy.io.loadmat('Data/cylinder_nektar_wake.mat')
X_star = data['X_star']
U = data['U_star']
V = U[:,1,:]
U = U[:,0,:]
P = data['p_star']
T = data['t']
del data


# %%



# %%
def pressureplot(P):
    vmin = np.min(np.abs(P[:,0:50]))
    vmax = np.max(np.abs(P[:,0:50]))
    Time = ['0.00','1.00','2.00','3.00','4.00','5.00']
    X = np.linspace(1,8,100)
    Y = np.linspace(-2,2,50)
    X,Y = np.meshgrid(X,Y)
    fig, ax = plt.subplots(3,2, figsize=(18,16))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    ax = ax.flatten()
    for i in range(6):
        p = P[:,i*10]
        p = np.reshape(p,(50,100))
        axcolor = ax[i].contour(X,Y,p,cmap=plt.cm.rainbow,levels=20)
        ax[i].set_xlabel("$x$",fontsize=16)
        ax[i].set_ylabel("$y$",fontsize=16)
        ax[i].set_title(u"涡度在t="+Time[i]+u"时的分布",fontsize=16)
        cb = fig.colorbar(axcolor,ax=ax[i])
        cb.ax.tick_params(labelsize='large')


# %%
def errorplot(P):
    vmin = np.min(np.abs(P[:,0:50]))
    vmax = np.max(np.abs(P[:,0:50]))
    Time = ['0.00','1.00','2.00','3.00','4.00','5.00']
    X = np.linspace(1,8,100)
    Y = np.linspace(-2,2,50)
    X,Y = np.meshgrid(X,Y)
    fig, ax = plt.subplots(3,2, figsize=(18,16))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    ax = ax.flatten()
    for i in range(6):
        p = P[:,i*10]
        p = np.reshape(p,(50,100))
        axcolor = ax[i].contourf(X,Y,p,cmap=plt.cm.plasma,levels=6)
        ax[i].set_xlabel("$x$",fontsize=16)
        ax[i].set_ylabel("$y$",fontsize=16)
        ax[i].set_title(u"压强在t="+Time[i]+u"时的分布",fontsize=16)
        cb = fig.colorbar(axcolor,ax=ax[i])
        cb.ax.tick_params(labelsize='large')


# %%
import scipy.io as io 
data = io.loadmat('Vorticity.mat')


# %%
VORTICITY = data['vorticity']


# %%
X = np.linspace(1,8,100)
Y = np.linspace(-2,2,50)
for i in range(51):
    p = VORTICITY[:,i]
    p = np.reshape(p,(50,100))
    fig = plt.figure(figsize=(9,5),dpi=100)
    plt.contour(X,Y,p,cmap=plt.cm.rainbow,levels=20)
    plt.xlabel("$x$",fontsize=16)
    plt.ylabel("$y$",fontsize=16)
    plt.title(u"涡度在"+'$t=$'+format('%.2f' %(0.1*i))+u"时的分布",fontsize=16)
    plt.colorbar()
    plt.savefig(fname='VORTICITY/vorticity_'+str(i))


# %%
X = np.linspace(1,8,100)
Y = np.linspace(-2,2,50)
for i in range(51):
    p = P[:,i]
    p = np.reshape(p,(50,100))
    fig = plt.figure(figsize=(9,5),dpi=100)
    plt.contourf(X,Y,p,cmap=plt.cm.rainbow,levels=20)
    plt.xlabel("$x$",fontsize=16)
    plt.ylabel("$y$",fontsize=16)
    plt.title(u"压强在"+'$t=$'+format('%.2f' %(0.1*i))+u"时的分布",fontsize=16)
    plt.colorbar()
    plt.savefig(fname='PRESSURE/pressure_'+str(i))


# %%
P = io.loadmat('approximate_P.mat')


# %%
P = P['approximate_P']


# %%
pressureplot(approximate_P)


# %%
pressureplot(VORTICITY)


# %%
errorplot(approximate_P - P[:,0:51])


# %%
def vorticity(model,x,y,t):
    with tf.GradientTape(persistent=True) as tape3:            
        Psi_u = tf.reshape(model(tf.concat([x,y,t],1))[:,0],[n_use,1])
        Psi_v = tf.reshape(model(tf.concat([x,y,t],1))[:,1],[n_use,1])
        Psi_p = tf.reshape(model(tf.concat([x,y,t],1))[:,2],[n_use,1])
    Psi_v_x = tape3.gradient(Psi_v,x)
    Psi_u_y = tape3.gradient(Psi_u,y)
    return Psi_v_x-Psi_u_y


# %%
VORTICITY = np.zeros((5000,51))
xyt_test[:,0:2] = X_star
n_use = 5000
for i in range(51):
    xyt_test[:,2] = i*0.1
    x = tf.Variable(tf.reshape(tf.cast(xyt_test[:,0], dtype=tf.float32),[n_use,1]))
    y = tf.Variable(tf.reshape(tf.cast(xyt_test[:,1], dtype=tf.float32),[n_use,1]))
    t = tf.Variable(tf.reshape(tf.cast(xyt_test[:,2], dtype=tf.float32),[n_use,1]))
    VORTICITY[:,i] =  np.reshape(vorticity(model=model,x=x,y=y,t=t).numpy(),[5000,])


# %%
# 生成初值点
t = np.zeros((5000,1))
xyt = np.hstack((X_star,t))
x_train_ic = np.reshape(xyt[:,0],[xyt.shape[0],1])
y_train_ic = np.reshape(xyt[:,1],[xyt.shape[0],1])
t_train_ic = np.reshape(xyt[:,2],[xyt.shape[0],1])
ic_train_u = U[:,0]
ic_train_v = V[:,0]

# 生成边界点
x_1 = np.ones((50*51,1))
y = np.linspace(-2,2,50)
t = np.linspace(0,5,51)
yt = np.zeros((50*51,2))
for i in range(50):
    for j in range(51):
        yt[51*i+j,0] = y[i]
        yt[51*i+j,1] = t[j]
x_1 = np.hstack((x_1,yt))
x_2 = np.ones((50*51,1))+7
x_2 = np.hstack((x_2,yt))

x = np.linspace(1,8,100)
t = np.linspace(0,5,51)
xt = np.zeros((100*51,3))
for i in range(100):
    for j in range(51):
        xt[51*i+j,0] = x[i]
        xt[51*i+j,2] = t[j]
y_1 = xt
y_1[:,1] = -2

xt1 = np.zeros((100*51,3))
for i in range(100):
    for j in range(51):
        xt1[51*i+j,0] = x[i]
        xt1[51*i+j,2] = t[j]
y_2 = xt1
y_2[:,1] = 2
del xt,xt1
xyt = np.vstack((x_1,x_2,y_1,y_2))
x_train_bc = np.reshape(xyt[:,0],[xyt.shape[0],1])
y_train_bc = np.reshape(xyt[:,1],[xyt.shape[0],1])
t_train_bc = np.reshape(xyt[:,2],[xyt.shape[0],1])
bc_train_u = np.hstack((np.reshape(U[np.where(X_star[:,0]==1),0:51],(50*51)),
                        np.reshape(U[np.where(X_star[:,0]==8),0:51],(50*51)),
                        np.reshape(U[np.where(X_star[:,1]==-2),0:51],(100*51)),
                        np.reshape(U[np.where(X_star[:,1]==2),0:51],(100*51))))                   
bc_train_v = np.hstack((np.reshape(V[np.where(X_star[:,0]==1),0:51],(50*51)),
                        np.reshape(V[np.where(X_star[:,0]==8),0:51],(50*51)),
                        np.reshape(V[np.where(X_star[:,1]==-2),0:51],(100*51)),
                        np.reshape(V[np.where(X_star[:,1]==2),0:51],(100*51)))) 
# 生成内部训练点
n_use = 100000
x_train = np.reshape(np.random.random(n_use)*7+1,[n_use,1])
y_train = np.reshape(np.random.random(n_use)*4-2,[n_use,1])
t_train = np.reshape(np.random.random(n_use)*5,[n_use,1])


# %%
# 训练集准备
x = tf.Variable(tf.reshape(tf.cast(x_train, dtype=tf.float32),[n_use,1]))
y = tf.Variable(tf.reshape(tf.cast(y_train, dtype=tf.float32),[n_use,1]))
t = tf.Variable(tf.reshape(tf.cast(t_train, dtype=tf.float32),[n_use,1]))


# %%
# 定义网络
struct = [3,50,50,50,50,3]
# 定义优化器
optimizer = Adam(learning_rate=0.001)
tf.keras.backend.set_floatx("float32")
model = keras.models.Sequential()
# 采用Xavier初始化方法
for layer in range(len(struct)-1):
    val = tf.cast(tf.sqrt(6/(struct[layer]+struct[layer+1])),dtype=tf.float32)
    if layer == len(struct)-2:
        model.add(keras.layers.Dense(units=struct[layer+1],kernel_initializer=initializers.random_uniform(-val,val)))
    elif layer == 0:
        model.add(keras.layers.Dense(units=struct[layer+1],input_dim=struct[layer],activation='tanh',
                                     kernel_initializer=initializers.random_uniform(-val,val)))
    else:
        model.add(keras.layers.Dense(units=struct[layer+1],activation='tanh',
                                     kernel_initializer=initializers.random_uniform(-val,val)))


# %%
# 定义残差函数(导数增长损失函数)
def residual(Psi_u,Psi_u_t,Psi_u_x,Psi_u_xx,Psi_u_y,Psi_u_yy,Psi_v,Psi_v_t,Psi_v_x,Psi_v_xx,Psi_v_y,Psi_v_yy,Psi_p_x,Psi_p_y):
    loss_d_1 = Psi_u_t + Psi_u*Psi_u_x + Psi_v*Psi_u_y + Psi_p_x - 0.01*(Psi_u_xx + Psi_u_yy)
    loss_d_2 = Psi_v_t + Psi_u*Psi_v_x + Psi_v*Psi_v_y + Psi_p_y - 0.01*(Psi_v_xx + Psi_v_yy)
    loss_d_3 = Psi_u_x + Psi_v_y
    loss_e = tf.reduce_mean(loss_d_1**2 + loss_d_2**2 + loss_d_3**2)
    return loss_e

# 定义边界损失函数
def loss_bc(model,x,y,t,bc_train_u,bc_train_v):
    loss_b = tf.reduce_mean((model(tf.concat([x,y,t],1))[:,0] - bc_train_u)**2 
                            + (model(tf.concat([x,y,t],1))[:,1] - bc_train_v)**2)
    return loss_b

# 定义初值损失函数
def loss_ic(model,x,y,t,ic_train_u,ic_train_v):
    loss_i = tf.reduce_mean((model(tf.concat([x,y,t],1))[:,0] - ic_train_u)**2 
                            + (model(tf.concat([x,y,t],1))[:,1] - ic_train_v)**2)
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
EPOCHS = 100 # 训练次数
alpha = 1 # 边界权重初始化
beta = 1
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
                Psi_u = tf.reshape(model(tf.concat([x,y,t],1))[:,0],[n_use,1])
                Psi_v = tf.reshape(model(tf.concat([x,y,t],1))[:,1],[n_use,1])
                Psi_p = tf.reshape(model(tf.concat([x,y,t],1))[:,2],[n_use,1])
            Psi_u_x, Psi_u_y, Psi_u_t = tape3.gradient(Psi_u,[x,y,t])
            Psi_v_x, Psi_v_y, Psi_v_t = tape3.gradient(Psi_v,[x,y,t])
            Psi_p_x, Psi_p_y = tape3.gradient(Psi_p,[x,y])
        Psi_u_xx = tape2.gradient(Psi_u_x,x)
        Psi_u_yy = tape2.gradient(Psi_u_y,y)
        Psi_v_xx = tape2.gradient(Psi_v_x,x)
        Psi_v_yy = tape2.gradient(Psi_v_y,y)
        loss_e = residual(Psi_u=Psi_u,Psi_u_t=Psi_u_t,Psi_u_x=Psi_u_x,Psi_u_xx=Psi_u_xx,Psi_u_y=Psi_u_y,Psi_u_yy=Psi_u_yy,
                          Psi_v=Psi_v,Psi_v_t=Psi_v_t,Psi_v_x=Psi_v_x,Psi_v_xx=Psi_v_xx,Psi_v_y=Psi_v_y,Psi_v_yy=Psi_v_yy,
                          Psi_p_x=Psi_p_x,Psi_p_y=Psi_p_y)
        loss_b = loss_bc(model=model,x=x_train_bc,y=y_train_bc,t=t_train_bc,bc_train_u=bc_train_u,bc_train_v=bc_train_v)
        loss_i = loss_ic(model=model,x=x_train_ic,y=y_train_ic,t=t_train_ic,ic_train_u=ic_train_u,ic_train_v=ic_train_v)
        loss = loss_e + alpha*loss_b + beta*loss_i
    if ((epoch+1)%100 == 0) and (method != 0): # 每100次更新一次权重alpha
        loss_e_gradients = tape1.gradient(loss_e,model.variables)
        loss_b_gradients = tape1.gradient(loss_b,model.variables)
        loss_i_gradients = tape1.gradient(loss_i,model.variables)
        alpha = alpha_estimate(a=alpha,loss_e_gradients=loss_e_gradients,loss_b_gradients=loss_b_gradients,method=method)
        beta = beta_estimate(b=beta,loss_e_gradients=loss_e_gradients,loss_i_gradients=loss_i_gradients,method=method)
        alpha_list.append(alpha)
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



# %%



# %%



# %%



