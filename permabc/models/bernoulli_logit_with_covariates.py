"""
Mô hình Bernoulli-logit với biến giải thích thời tiết.

Mô-đun này triển khai một mô hình logistic phân cấp, trong đó:
- Tham số toàn cục (beta) mô tả tác động của các đặc trưng
- Tham số cục bộ (alpha_k) mô tả intercept riêng cho từng thành phần
- Dùng để ước lượng xác suất mưa từ các biến thời tiết
"""

import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
import numpy as np
from scipy.special import expit

# Import from package structure
try:
    from . import ModelBase
    from ..utils.functions import Theta
except ImportError:
    # Fallback for old structure
    try:
        from models import ModelBase
        from utils.functions import Theta
    except ImportError:
        from . import ModelBase
        from ..utils.functions import Theta


class BernoulliLogitWithCovariates(ModelBase):
    """
    Hierarchical Bernoulli-logit model with weather feature covariates.
    
    This model assumes:
    - α_k ~ Normal(μ_α, σ_α²) for each component k (intercepts)
    - β_j ~ Normal(μ_β, σ_β²) for each feature j (global effects)
    - y_k,i | x_k,i ~ Bernoulli(logit^{-1}(α_k + x_k,i^T β))
    
    The model is designed for binary rain probability estimation across multiple
    geographic regions/provinces, using normalized weather features as covariates.
    
    Parameters
    ----------
    K : int
        Number of components (regions/provinces).
    n_obs : int, default=1
        Number of observations per component.
    n_features : int, default=5
        Number of weather features (covariates).
    mu_alpha : float, default=0
        Prior mean for component intercepts α_k.
    sigma_alpha : float, default=2
        Prior standard deviation for component intercepts α_k.
    mu_beta : float, default=0
        Prior mean for feature coefficients β_j.
    sigma_beta : float, default=2
        Prior standard deviation for feature coefficients β_j.
    X_cov : np.ndarray, optional
        Covariate matrix of shape (K, n_obs, n_features).
        If None, will use uniform covariates.
    """
    
    def __init__(
        self,
        K,
        n_obs=1,
        n_features=5,
        mu_alpha=0.0,
        sigma_alpha=2.0,
        mu_beta=0.0,
        sigma_beta=2.0,
        X_cov=None,
    ):
        """Initialize the Bernoulli-logit model with covariates."""
        super().__init__(K)
        
        # Model parameters
        self.n_obs = n_obs
        self.n_features = n_features
        self.mu_alpha = mu_alpha
        self.sigma_alpha = sigma_alpha
        self.mu_beta = mu_beta
        self.sigma_beta = sigma_beta
        
        # Covariate matrix: store as provided (already normalized externally)
        if X_cov is None:
            X_cov = np.ones((K, n_obs, n_features))
        self.X_cov = np.asarray(X_cov, dtype=np.float32)
        
        # Verify shape consistency
        if self.X_cov.shape != (K, n_obs, n_features):
            raise ValueError(
                f"X_cov shape {self.X_cov.shape} does not match expected (K={K}, n_obs={n_obs}, n_features={n_features})"
            )
        
        # Miền giá trị tham số
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])  # alpha_k có thể nhận mọi giá trị thực
        self.support_par_glob = jnp.array([[-jnp.inf, jnp.inf]] * n_features)  # beta_j có thể nhận mọi giá trị thực

        # Kích thước tham số và tên hiển thị
        self.dim_loc = 1  # alpha_k là một vô hướng cho mỗi thành phần
        self.dim_glob = n_features  # beta_j là một vector theo số đặc trưng
        self.loc_name = ["$\\alpha_{"]  # Tên LaTeX cho tham số cục bộ
        self.glob_name = [f"$\\beta_{{{i}}}$" for i in range(n_features)]  # Tên LaTeX cho hệ số đặc trưng
    
    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Sinh mẫu từ phân phối prior.

        Dùng NumPy random để tránh JAX recompilation khi số particles thay đổi
        giữa các vòng SMC.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Khóa sinh số ngẫu nhiên.
        n_particles : int
            Số particle cần sinh.
        n_silos : int, default=0
            Số thành phần (nếu 0 thì mặc định bằng K).

        Returns
        -------
        Theta
            Mẫu tham số với loc=(n_particles, K, 1) cho alpha_k
            và glob=(n_particles, n_features) cho beta_j.
        """
        if n_silos == 0:
            n_silos = self.K
        
        rng = np.random.default_rng(int(key[0]))
        
        # Lấy mẫu intercept alpha_k ~ Normal(mu_alpha, sigma_alpha^2)
        alphas = (
            rng.standard_normal((n_particles, n_silos, 1)) * self.sigma_alpha + self.mu_alpha
        )
        
        # Lấy mẫu hệ số beta_j ~ Normal(mu_beta, sigma_beta^2) cho từng đặc trưng
        betas = (
            rng.standard_normal((n_particles, self.n_features)) * self.sigma_beta + self.mu_beta
        )
        
        return Theta(loc=alphas, glob=betas)

    def prior_generator_jax(self, key, n_particles, n_silos=0):
        """Phiên bản prior_generator dùng JAX."""
        if n_silos == 0:
            n_silos = self.K

        key, key_alpha, key_beta = random.split(key, 3)
        alphas = random.normal(key_alpha, shape=(n_particles, n_silos, 1)) * self.sigma_alpha + self.mu_alpha
        betas = random.normal(key_beta, shape=(n_particles, self.n_features)) * self.sigma_beta + self.mu_beta
        return alphas, betas
    
    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior distribution.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values with loc=(n_particles, K, 1) and glob=(n_particles, n_features).
            
        Returns
        -------
        np.ndarray
            Log prior densities of shape (n_particles,).
        """
        # Log pdf for α_k ~ Normal(μ_α, σ_α²)
        log_pdf_alpha = norm.logpdf(
            np.asarray(thetas.loc), loc=self.mu_alpha, scale=self.sigma_alpha
        )
        log_pdf_alpha_sum = np.sum(log_pdf_alpha, axis=(1, 2))
        
        # Log pdf for β_j ~ Normal(μ_β, σ_β²)
        log_pdf_beta = norm.logpdf(
            np.asarray(thetas.glob), loc=self.mu_beta, scale=self.sigma_beta
        )
        log_pdf_beta_sum = np.sum(log_pdf_beta, axis=1)
        
        return log_pdf_alpha_sum + log_pdf_beta_sum
    
    def data_generator(self, key, thetas):
        """
        Sinh quan sát nhị phân mô phỏng từ mô hình logit.

        Với mỗi thành phần k và mỗi quan sát i, ta sinh y_k,i từ
        Bernoulli(sigmoid(alpha_k + X_cov[k,i,:] · beta)).

        Dùng NumPy để sinh ngẫu nhiên nhằm tránh JAX JIT recompilation.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Khóa sinh số ngẫu nhiên.
        thetas : Theta
            Mẫu tham số với loc=(n_particles, K, 1) và glob=(n_particles, n_features).

        Returns
        -------
        np.ndarray
            Quan sát nhị phân mô phỏng dạng (n_particles, K, n_obs).
        """
        n_particles = thetas.loc.shape[0]
        n_silos = thetas.loc.shape[1]
        
        alphas = np.asarray(thetas.loc)  # (n_particles, K, 1)
        betas = np.asarray(thetas.glob)  # (n_particles, n_features)
        
        # Create RNG
        rng = np.random.default_rng(int(key[0]))

        # Over-sampling có thể yêu cầu nhiều thành phần mô phỏng hơn số thành phần quan sát.
        # Lặp vòng template covariate để model vẫn sinh được M > K mà không đổi API ngoài.
        if n_silos == self.K:
            X_cov = self.X_cov
        else:
            component_ids = np.arange(n_silos) % self.K
            X_cov = np.take(self.X_cov, component_ids, axis=0)
        
        # Tính linear predictor: eta_k,i = alpha_k + X_cov[k,i,:] · beta
        # alphas: (n_particles, K, 1)
        # betas: (n_particles, n_features)
        # X_cov: (K, n_obs, n_features)
        
        # Mở rộng chiều để broadcast:
        # alphas -> (n_particles, K, 1)
        # X_cov -> (1, K, n_obs, n_features)
        # betas -> (n_particles, 1, 1, n_features)
        
        X_expanded = X_cov[np.newaxis, :, :, :]  # (1, n_silos, n_obs, n_features)
        betas_expanded = betas[:, np.newaxis, np.newaxis, :]  # (n_particles, 1, 1, n_features)
        alphas_expanded = alphas[:, :, 0]  # (n_particles, n_silos) - extract scalar from last dim
        
        # Tính phần đóng góp từ đặc trưng: (n_particles, 1, 1, n_features) * (1, K, n_obs, n_features)
        # Kết quả: (n_particles, K, n_obs, n_features), rồi cộng theo trục cuối -> (n_particles, K, n_obs)
        feature_contribution = np.sum(X_expanded * betas_expanded, axis=-1)  # (n_particles, K, n_obs)
        
        # Cộng intercept: (n_particles, n_silos) + (n_particles, n_silos, n_obs)
        eta = alphas_expanded[:, :, np.newaxis] + feature_contribution  # (n_particles, K, n_obs)
        probs = expit(eta)  # (n_particles, K, n_obs)
        
        # Sinh quan sát nhị phân
        zs = rng.binomial(1, probs)  # (n_particles, K, n_obs)
        
        return zs.astype(np.float32)

    def data_generator_jax(self, key, thetas_loc, thetas_glob):
        """Phiên bản data_generator dùng JAX."""
        n_particles = thetas_loc.shape[0]
        n_silos = thetas_loc.shape[1]

        if n_silos == self.K:
            X_cov = jnp.asarray(self.X_cov)
        else:
            component_ids = jnp.arange(n_silos) % self.K
            X_cov = jnp.take(jnp.asarray(self.X_cov), component_ids, axis=0)

        feature_contribution = jnp.sum(
            X_cov[jnp.newaxis, :, :, :] * thetas_glob[:, jnp.newaxis, jnp.newaxis, :],
            axis=-1,
        )
        eta = thetas_loc[:, :, 0][:, :, jnp.newaxis] + feature_contribution
        probs = jax.nn.sigmoid(eta)
        key, key_data = random.split(key)
        zs = random.bernoulli(key_data, probs).astype(jnp.float32)
        return zs
    
    def distance_matrices_loc(self, zs, y_obs, M=0, L=0):
        """
        Tính ma trận khoảng cách cục bộ cho dữ liệu nhị phân.

        Dùng Hamming distance (số vị trí khác nhau) làm chi phí ghép
        giữa thành phần mô phỏng và thành phần quan sát.

        Parameters
        ----------
        zs : np.ndarray
            Dữ liệu mô phỏng dạng (n_particles, M, n_obs).
        y_obs : np.ndarray
            Dữ liệu quan sát dạng (1, K, n_obs).
        M : int, default=0
            Số thành phần mô phỏng (mặc định bằng K).
        L : int, default=0
            Số thành phần cần ghép (mặc định bằng K).

        Returns
        -------
        np.ndarray
            Ma trận khoảng cách dạng (n_particles, K, M), trong đó
            các hàng là thành phần quan sát và các cột là thành phần mô phỏng.
        """
        if M == 0:
            M = self.K
        if L == 0:
            L = self.K
        
        n_particles = zs.shape[0]
        
        # Lấy dữ liệu quan sát cho K thành phần đầu
        y_obs_data = np.asarray(y_obs[0, :self.K, :])  # (K, n_obs)
        
        # Tính Hamming distance theo từng thành phần.
        # Mỗi lỗi lệch được trọng số bởi weights_distance của thành phần quan sát.
        distances = np.zeros((n_particles, self.K, M), dtype=np.float32)
        comp_weights = np.asarray(self.weights_distance[:self.K], dtype=np.float32)
        
        for p in range(n_particles):
            for j in range(self.K):
                for i in range(M):
                    # Hamming distance: số vị trí mà giá trị nhị phân khác nhau
                    mismatch_count = np.sum(np.abs(zs[p, i, :] - y_obs_data[j, :]))
                    distances[p, j, i] = comp_weights[j] ** 2 * mismatch_count
        
        return distances
    
    def distance_global(self, zs, y_obs):
        """
        Tính khoảng cách toàn cục cho tham số.

        Với mô hình này, không có thành phần global distance riêng ngoài
        phần ghép cục bộ, nên trả về vector 0.
        """
        n_particles = zs.shape[0]
        return np.zeros(n_particles, dtype=np.float32)
    
    def distance(self, zs, y_obs):
        """
        Tính khoảng cách tổng giữa dữ liệu mô phỏng và quan sát.

        Hàm này có hoán vị component bằng Hungarian để phản ánh đúng
        cấu trúc exchangeable giữa các thành phần.
        """
        y_obs_data = np.asarray(y_obs[0, :self.K, :])  # (K, n_obs)
        n_particles = zs.shape[0]
        comp_weights = np.asarray(self.weights_distance[:self.K], dtype=np.float32)

        distances = np.zeros(n_particles, dtype=np.float32)
        for p in range(n_particles):
            cost = np.zeros((self.K, self.K), dtype=np.float32)
            for j in range(self.K):
                for i in range(self.K):
                    mismatch_count = np.sum(np.abs(zs[p, i, :] - y_obs_data[j, :]))
                    cost[j, i] = (comp_weights[j] ** 2) * mismatch_count

            row_ind, col_ind = linear_sum_assignment(cost)
            distances[p] = np.sqrt(np.sum(cost[row_ind, col_ind]))

        return distances
    
    def summary(self, z):
        """
        Áp dụng biến đổi summary cho dữ liệu thô.

        Với dữ liệu nhị phân, tóm tắt bằng tỷ lệ mưa theo từng thành phần.
        """
        z = jnp.asarray(z)
        return jnp.mean(z, axis=-1, keepdims=True)
    
    def set_X_cov(self, X_cov):
        """
        Cập nhật ma trận covariate sau khi khởi tạo.

        Hữu ích khi covariate được tính sau lúc tạo model.
        """
        X_cov = np.asarray(X_cov, dtype=np.float32)
        if X_cov.shape != (self.K, self.n_obs, self.n_features):
            raise ValueError(
                f"X_cov shape {X_cov.shape} does not match expected "
                f"(K={self.K}, n_obs={self.n_obs}, n_features={self.n_features})"
            )
        self.X_cov = X_cov
