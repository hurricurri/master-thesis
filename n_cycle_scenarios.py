import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def def_states(n):

    cos_theta = np.sqrt(np.cos(np.pi/n)/(1+np.cos(np.pi/n)))
    sin_theta = np.sqrt( 1-cos_theta**2 )
    phi_1 = np.pi*(n-1)/n


    global cycle_states
    cycle_states = [np.array([cos_theta, sin_theta*np.sin(phi_1*i), sin_theta*np.cos(phi_1*i)]) for i in range(1,n+1)]
    
    return


def project(j, outcome, states):

    dot_products = np.einsum("i,ji->j",cycle_states[j-1],states)

    
    if outcome == 1:
        phases = np.absolute(dot_products)
        states = cycle_states[j-1] * phases[:,None]
        return states
        
    if outcome == 0:
        states = states - cycle_states[j-1] * dot_products[:,None]
        phases = np.sign(states[:,0])
        states = states * phases[:,None]
        return states

        

def new_mnt(old_states, old_probs):
    
    new_states = np.array([])
    new_probs = np.array([])
    
    for mnt in range(1,n+1):
        for outcome in range(2): 
        
            upd_states = project(mnt, outcome, old_states)
            
            norms = np.linalg.norm(upd_states, axis = 1)
            upd_probs = np.multiply(old_probs,(norms**2/n))

            zero_idx = np.where(upd_probs == 0)[0]
            upd_probs = np.delete(upd_probs, zero_idx)
            
            upd_states = np.delete(upd_states, zero_idx, axis = 0)
            norms = np.delete(norms,zero_idx)
                        
            upd_states = upd_states/norms[:,None]
            
            if mnt == 1 and outcome == 0:
                new_states = upd_states
                new_probs = upd_probs
                continue
            
            for row_idx, row in enumerate(upd_states):
                # find duplicates
                dup_idx = row_in_matrix(new_states, row)
                
                if len(dup_idx):
                
                    new_probs[dup_idx[0]] += upd_probs[row_idx]
                    continue
                
                new_states = np.row_stack((new_states,row))
                new_probs = np.append(new_probs,upd_probs[row_idx])
    
    return new_states, new_probs



def row_in_matrix(A,x):
    bool_array = np.isclose(A,x)
    return np.where(np.all(bool_array,axis=1))[0]

def entry_in_array(A,x):
    bool_array = np.isclose(A,x)
    return np.where(bool_array)[0]

def simulate(n_array,no_mnts,plot_states=False):
    entropies = np.zeros((len(n_array),no_mnts))
    erased_bits = np.zeros(len(n_array))
    
    
    for i,cycle_len in enumerate(n_array):
        global n
        n = cycle_len
        def_states(n)

        states = np.array([[1,0,0]])
        probs = np.array([1])
    
        
        
        for k in range(no_mnts):
            print(k+1)
            states, probs = new_mnt(states, probs) 

            
            entropy = -np.sum(np.multiply(probs,np.log2(probs)))
            entropies[i,k] = entropy
            
            if plot_states:
                fig = plt.figure(figsize=(10,5))
                ax = fig.gca(projection='3d')

                ax.set_xlim3d(-1, 1)
                ax.set_ylim3d(-1, 1)
                ax.set_zlim3d(-1, 1)
                
                # Get rid of the panes
                ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

                # Get rid of the spines
                ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

                # Get rid of the ticks
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                ax.set_zticks([])
                
                ax.scatter(*states.T, s= 5000*probs, marker = "." ,color="black")
                ax.view_init(0, 0)
                plt.savefig("n = " + str(n) + ", mnts = " + str(k+1), dpi=500)

        erased = 0
        for mnt in range(1,n+1):
            for outcome in range(2):
            
                squared_norms = np.linalg.norm(project(mnt, outcome, states),axis=1)**2

                
                to_sum = np.multiply( np.multiply(squared_norms,np.log2(squared_norms/n)), probs)
                to_sum = to_sum[~np.isnan(to_sum)]

                erased -= 1/n*np.sum(to_sum)
                
        erased_bits[i] = erased
        
        
    fig = plt.figure(figsize=(8,6))
    
    ax = fig.gca()

   
    plt.xticks(np.log2(n_array),('$log_2$(5)','$log_2$(7)','$log_2$(9)','$log_2$(11)','$log_2$(13)','$log_2$(15)'))
    plt.yticks(np.array([2.5,3,3.5,4,4.5,5]))
    plt.xlabel("$log_2$(cycle length)", labelpad=10)
    plt.ylabel("erased bits", labelpad=10)
    plt.grid(linestyle='--',color="lightgrey")
    
    plt.plot(np.log2(n_array), erased_bits, "o", color = "black")
    plt.savefig("erased")
    

    fig = plt.figure()
    plt.xlabel("number of measurements", labelpad=10)
    plt.ylabel("entropy H", labelpad=10)
    plt.xticks(range(1,no_mnts+1))
    plt.grid(linestyle='--',color="lightgrey")
    
    markers = ["o" , "s" , "^", "D"]
    colours = ["black" , "white" , "black", "darkgrey"]
    edgecolours = ["black" , "black" , "black", "dimgrey"]
    
    
    for i,row in enumerate(entropies[0:4]):
        plt.scatter(np.arange(1,no_mnts+1),entropies[i], marker=markers[i] , color=colours[i], edgecolors=edgecolours[i] , s =15, label = "n = "+ str(n_array[i]))
    
    plt.legend()
   
    
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()

    
    plt.xticks(np.log2(n_array),('$log_2$(5)','$log_2$(7)','$log_2$(9)','$log_2$(11)','$log_2$(13)','$log_2$(15)'))
    #plt.yticks()
    plt.xlabel("$log_2$(cycle length)", labelpad=10)
    plt.ylabel("memory cost (bits)", labelpad=10)
    
    plt.plot(np.log2(n_array), entropies[:,-1],"o", color = "black")
    
    #coeffs = np.polyfit(np.log(np.array(n_array)), entropies[:,-1] , 1)
    #plt.plot(np.linspace(4.5,n_array[-1]+0.5,100), coeffs[1] + coeffs[0]*np.log(np.linspace(4.5,n_array[-1]+0.5,100)), "--", color = "darkgrey")
    plt.grid(linestyle='--',color="lightgrey")
    plt.savefig("n_cycle_RAM")
    
        
simulate(np.array([5,7,9,11,13,15]),3,False)

#simulate(np.array([5,7]),5,False)
plt.show()



    
