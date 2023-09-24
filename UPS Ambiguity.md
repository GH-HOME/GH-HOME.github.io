#The ambigulties in uncalibrated photometric stereo

##Problem setting
Let's denote light direction as L, albedo as $\rho$, normal as N and light intensity of the image as I. If we focue on Lambertian reflectance model, then we can get that 
$$
I_{mn}=\rho N_{m,3} L_{3,n}\\
\rho=\|N\|\\
k=\|L\|
$$
here the subscript m and n represent the amounts of the pixels and the light respectively. For uncalibrated photometric stereo, we only know intensity image seguence under different light condition $L$. We want to determine the other items.

## Problem solving
1. We first conduct SVD on $I$,
$$
I_{mn}=U_{m,3} \Sigma_{3,3} V_{3,n}
$$
here, we fix the dimension of $\Sigma$ to be 3 since the rank of light and normal vector should be 3.

2. We suppose 
$$
\hat N=U_{m,3} \sqrt{\Sigma}\\
\hat L= \sqrt{\Sigma} V_{n,3}
$$
here we still cannot determine the ground truth normal and light since there will be a $3 \times 3$ ambigulties matrix $G$ including:
$$
I_{m,n} = \hat N G G^{-1} \hat L
$$

3. If albedos are known at 6 different normals or lighting intensity is known at 6 images, this linear ambiguity can be reduced to a rotational ambiguity. For example, if we know albedos at 6 different normal, then each normal can provide one equation:
$$
\|\hat N_i G \|=\rho_i \\
\hat N_i G G^T \hat N_i^T  =\rho_i
$$
Let's denote $K=GG^T$, Since K is a $3 \times 3 $ real symmetric matric, there will be only 6 unknown elements, therefore we can determine K with more than 6 different normals with albedo known. 
Next, we need to determine G from K. By symmetric matrix decomposition, we have
$$
K=L \Lambda L^T   
$$
here $L$ is orthogonal matrix with $L \cdot L^T=I$ and $\Lambda$ is a dignoal matrix with $\sqrt \Lambda^T= \sqrt\Lambda$, so we can further get
$$
K=L\sqrt\Lambda \cdot (L \sqrt \Lambda )^T
$$
So will $G=L\sqrt\Lambda$, the answer is No. For every G, we also have a orthogonal matrix to make $GRR^TG^T=K$, we after symmetric matrix decomposition, we can get 
$$GR=L\sqrt\Lambda$$
orthogonal matrix also called rotation matric, so after that we left rotation ambiguity.


