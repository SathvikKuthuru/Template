/******************************************************************************

                      This is Aeren's algorithm template                       
                         for competitive programming

******************************************************************************/
/******************************************************************************
Category


1. Number Theory
	1.1. Modular Exponentiation, Modular Inverse
		156485479_1_1
	1.2. Extended Euclidean Algorithm
		156485479_1_2
	1.3. Linear Sieve
		156485479_1_3
	1.4. Combinatorics
		156485479_1_4
	1.5. Euler Totient Function
		156485479_1_5
	1.6. Millar Rabin Primality Test
		156485479_1_6
	1.7. Pollard Rho and Factorization
		156485479_1_7
	1.8. Tonelli Shanks Algorithm ( Solution to x^2 = a mod p )
		156485479_1_8
	1.9. Chinese Remainder Theorem
		156485479_1_9
	1.10. Lehman Factorization
		156485479_1_10
	1.11. Mobius Function
		156485479_1_11


2. Numeric
	2.1. Linear Recurrence Relation Solver / Berlekamp Massey Algorithm
		156485479_2_1
	2.2. System of Linear Equations
		2.2.1. Coefficients in R
			156485479_2_2_1
		2.2.2. Coefficients in Z_p
			156485479_2_2_2
		2.2.3. Coefficients in Z_2
			156485479_2_2_3
	2.3. Matrix
		2.3.1. Entries in R
			156485479_2_3_1
		2.3.2. Entries in some ring
			156485479_2_3_2
	2.4. Polynomial
		2.4.1. Convolution
			2.4.1.1 Addition Convolution
				2.4.1.1.1. Fast Fourier Transform
					156485479_2_4_1_1_1
				2.4.1.1.2. Number Theoric Transform
					156485479_2_4_1_1_2
			2.4.1.2. Bitwise XOR Convolution ( Fast Walsh Hadamard Transform )
				156485479_2_4_1_2
			2.4.1.3. Bitwise AND Convolution
				156485479_2_4_1_3
			2.4.1.4. Bitwise OR Convolution
				156485479_2_4_1_4
		2.4.2. Interpolation
			2.4.2.1. Slow Interpolation
				2.4.2.1.1.
					156485479_2_4_2_1_1
				2.4.2.1.2.
					156485479_2_4_2_1_2
			2.4.2.2. Fast Interpolation
				156485479_2_4_2_2 ( INCOMPLETE )
	2.5. Kadane
		156485479_2_5
	2.6. DP Optimization
		2.6.1. Convex Hull Trick ( Line Containers / Li Chao Tree )
			2.6.1.1. Sorted Line Container
				156485479_2_6_1_1
			2.6.1.2. Line Container
				156485479_2_6_1_2
			2.6.1.3. Li Chao Tree
				156485479_2_6_1_3
		2.6.2. Divide and Conquer
			156485479_2_6_2
		2.6.3. Knuth
			156485479_2_6_3
		2.6.4. Lagrange ( Aliens Trick, Wqs Binary Search )
			156485479_2_6_4
	2.7. Binary Search
		156485479_2_7


3. Data Structure
	3.1. Sparse Table
		156485479_3_1
	3.2. Segment Tree
		3.2.1. Simple Iterative Segment Tree
			156485479_3_2_1
		3.2.2. Iterative Segment Tree with Reversed Operation
			156485479_3_2_2
		3.2.3. Iterative Segment Tree Supporting Lazy Propagation
			156485479_3_2_3 ( INCOMPLETE )
		3.2.4. Recursive Segment Tree
			156485479_3_2_4
		3.2.5. 2D Segment Tree
			156485479_3_2_5 ( INCOMPLETE )
		3.2.6. Lazy Dynamic Segment Tree
			156485479_3_2_6 
		3.2.7. Persistent Segment Tree
			156485479_3_2_7
	3.3. Fenwick Tree
		3.3.1. Simple Fenwick Tree
			156485479_3_3_1
		3.3.2. Fenwick Tree Supporting Range Queries of The Same Type
			156485479_3_3_2
		3.3.3. 2D Fenwick Tree
			156485479_3_3_3
	3.4. Wavelet Tree
		156485479_3_4 ( NOT THROUGHLY TESTED YET )
	3.5. Disjoint Set
		156485479_3_5
	3.6. Monotone Stack
		156485479_3_6
	3.7. Persistent Array
		156485479_3_8
	3.8. Persistent Disjoint Set
		156485479_3_9
	3.9. Less-than-k Query / Distinct Value Query
		156485479_3_11
	3.10. Mo's Algorithm
		156485479_3_12

4. Graph
	4.1. Strongly Connected Component ( Tarjan's Algorithm )
		156485479_4_1
	4.2. Biconnected Component
		156485479_4_2
	4.3. Flow Network
		4.3.1. Dinic's Maximum Flow Algorithm
			156485479_4_3_1
		4.3.2. Minimum Cost Maximum Flow Algorithm
			156485479_4_3_2
	4.4. Tree Algorithms
		4.4.1. LCA
			156485479_4_4_1
		4.4.2. Binary Lifting
			4.4.2.1. Unweighted Tree
				156485479_4_4_2_1
			4.4.2.2. Weighted Tree
				156485479_4_4_2_2
		4.4.3. Heavy Light Decomposition
			156485479_4_4_3
		4.4.4. Centroid Decomposition
			156485479_4_4_4
		4.4.5. AHU Algorithm ( Rooted Tree Isomorphism ) / Tree Isomorphism
			156485479_4_4_5

5. String
	5.1. Lexicographically Minimal Rotation
		156485479_5_1
	5.2. Palindromic Substrings ( Manacher's Algorithm )
		156485479_5_2
	5.3. Suffix Array and Kasai's Algorithm
		156485479_5_3
	5.4. Z Function
		156485479_5_4
	5.5. Aho Corasic
		156485479_5_5
	5.6. Prefix Function / Prefix Automaton
		156485479_5_6
	5.7. Polynomial Hash
		156485479_5_7


6. Geometry
	6.1. 2D Geometry
		156485479_6_1
	6.2. Convex Hull and Minkowski Addition
		156485479_6_2


7. Miscellaneous
	7.1. Custom Hash Function for unordered_set and unordered map
		156485479_7_1
	7.2. Bump Allocator
		156485479_7_2

*******************************************************************************/

// 156485479_1_1
// Modular Exponentiation, Modular Inverse and Geometric Sum
// O(log e)
ll modexp(ll b, ll e, const ll &mod){
	ll res = 1;
	for(; e; b = b * b % mod, e >>= 1) if(e & 1) res = res * b % mod;
	return res;
}
ll modinv(ll b, const ll &mod){
	return modexp(b, mod - 2, mod);
}
ll modgeo(ll b, ll e, const ll &mod){
	if(e < 2) return e;
	ll res = 1;
	for(ll bit = 1 << 30 - __builtin_clz(e), p = 1; bit; bit >>= 1){ // 0->1->2
		res = res * (1 + p * b % mod) % mod, p = p * p % mod * b % mod;
		if(bit & e) res = (res + (p = p * b % mod)) % mod;
	}
	return res;
}
template<class T>
T binexp(T b, ll e, const T &id){
	T res = id;
	for(; e; b = b * b, e /= 2) if(e & 1) res = res * b;
	return res;
}
template<class T>
T bingeo(const T &b, ll e, const T &add_id, const T &mul_id){
	if(e < 2) return e ? mul_id : add_id;
	T res = mul_id, p = mul_id;
	for(ll bit = 1 << 30 - __builtin_clz(e); bit; bit >>= 1){
		res = res * (mul_id + p * b), p = p * p * b;
		if(bit & e) res = (res + (p = p * b));
	}
	return res;
}

// 156485479_1_2
// Extended Euclidean Algorithm
// O(max(log x, log y))
ll euclid(ll x, ll y, ll &a, ll &b){
	if(y){
		ll d = euclid(y, x % y, b, a);
		return b -= x / y * a, d;
	}
	return a = 1, b = 0, x;
}

// 156485479_1_3
// Run linear sieve up to n
// O(n)
void linearsieve(int n, vi &lpf, vi &prime){
	lpf.resize(n + 1);
	prime.reserve(n + 1);
	for(int i = 2; i <= n; i ++){
		if(!lpf[i]){
			lpf[i] = i;
			prime.push_back(i);
		}
		for(int j = 0; j < prime.size() && prime[j] <= lpf[i] && i * prime[j] <= n; j ++){
			lpf[i * prime[j]] = prime[j];
		}
	}
}

// 156485479_1_4
// Combinatorics
// O(N) preprocessing, O(1) per query
struct combinatorics{
	const ll N, mod;
	vl inv, fact, invfact;
	combinatorics(ll N, ll mod):
		N(N), mod(mod), inv(N + 1), fact(N + 1), invfact(N + 1){
		inv[1] = 1, fact[0] = fact[1] = invfact[0] = invfact[1] = 1;
		for(ll i = 2; i <= N; i ++){
			inv[i] = (mod - mod / i * inv[mod % i] % mod) % mod;
			fact[i] = fact[i - 1] * i % mod;
			invfact[i] = invfact[i - 1] * inv[i] % mod;
		}
	}
	ll C(int n, int r){
		return n < r ? 0 : fact[n] * invfact[r] % mod * invfact[n - r] % mod;
	}
	ll P(int n, int r){
		return n < r ? 0 : fact[n] * invfact[n - r] % mod;
	}
	ll H(int n, int r){
		return c(n + r - 1, r);
	}
	ll Cat(int n, int k, int m){
		if(m <= 0) return 0;
		else if(k >= 0 && k < m) return c(n + k, k);
		else if(k < n + m) return (c(n + k, k) - c(n + k, k - m) + mod) % mod;
		else return 0;
	}
};

// 156485479_1_5
// Euler Totient Function
// O(sqrt(x))
ll phi(ll x){
	ll res = x;
	for(ll i = 2; i * i <= x; i ++){
		if(x % i == 0){
			while(x % i == 0) x /= i;
			res -= res / i;
		}
	}
	if(x > 1) res -= res / x;
	return res;
}
// Calculate phi(x) for all 1 <= x <= n
// O(n log n)
void process_phi(int n, vi &phi){
	phi.resize(n);
	for(int i = 0; i <= n; i ++){
		phi[i] = i & 1 ? i : i / 2;
	}
	for(int i = 3; i <= n; i += 2){
		if(phi[i] == i){
			for(int j = i; j <= n; j += i){
				phi[j] -= phi[j] / i;
			}
		}
	}
}

// 156485479_1_6
// Millar Rabin Primality Test
// O(log n) {constant is around 7}
typedef unsigned long long ull;
typedef long double ld;
ull mod_mul(ull a, ull b, ull M) {
	ll res = a * b - M * ull(ld(a) * ld(b) / ld(M));
	return res + M * (res < 0) - M * (res >= (ll)M);
}
ull mod_pow(ull b, ull e, ull mod) {
	ull res = 1;
	for (; e; b = mod_mul(b, b, mod), e /= 2) if (e & 1) res = mod_mul(res, b, mod);
	return res;
}
bool isprime(ull n){
	if(n < 2 || n % 6 % 4 != 1) return n - 2 < 2;
	vector<ull> A{2, 325, 9375, 28178, 450775, 9780504, 1795265022};
	ull s = __builtin_ctzll(n - 1), d = n >> s;
	for(auto a: A){
		ull p = mod_pow(a, d, n), i = s;
		while(p != 1 && p != n - 1 && a % n && i --) p = mod_pow(p, p, n);
		if(p != n - 1 && i != s) return 0;
	}
	return 1;
}

// 156485479_1_7
// Pollard Rho Algorithm
// O(n^{1/4} log n)
ull pfactor(ull n){
	auto f = [n](ull x){
		return (mod_mul(x, x, n) + 1) % n;
	};
	if(!(n & 1)) return 2;
	for(ull i = 2; ; i ++){
		ull x = i, y = f(x), p;
		while((p = gcd(n + y - x, n)) == 1){
			x = f(x), y = f(f(y));
		}
		if(p != n){
			return p;
		}
	}
}
vector<ull> factorize(ull n){
	if(n == 1){
		return {};
	}
	if(isprime(n)){
		return {n};
	}
	ull x = pfactor(n);
	auto l = factorize(x), r = factorize(n / x);
	l.insert(l.end(), all(r));
	return l;
}

// 156485479_1_8
// Tonelli Shanks Algorithm ( Solution to x^2 = a mod p )
// O(log^2 p)
ll modexp(ll b, ll e, const ll &mod){
	ll res = 1;
	for(; e; b = b * b % mod, e >>= 1) if(e & 1) res = res * b % mod;
	return res;
}
ll modinv(ll b, const ll &mod){
	return modexp(b, mod - 2, mod);
}
ll sqrt(ll a, ll p){
	a %= p;
	if(a < 0) a += p;
	if(a == 0) return 0;
	assert(modexp(a, (p - 1)/2, p) == 1);
	if(p % 4 == 3) return modexp(a, (p+1)/4, p);
	// a^(n+3)/8 or 2^(n+3)/8 * 2^(n-1)/4 works if p % 8 == 5
	ll s = p - 1, n = 2;
	int r = 0, m;
	while(s % 2 == 0) ++ r, s /= 2;
	/// find a non-square mod p
	while(modexp(n, (p - 1) / 2, p) != p - 1) ++ n;
	ll x = modexp(a, (s + 1) / 2, p);
	ll b = modexp(a, s, p), g = modexp(n, s, p);
	for(;; r = m){
		ll t = b;
		for(m = 0; m < r && t != 1; ++ m) t = t * t % p;
		if(m == 0) return x;
		ll gs = modexp(g, 1LL << (r - m - 1), p);
		g = gs * gs % p;
		x = x * gs % p;
		b = b * g % p;
	}
}

// 156485479_1_9
// Chinese Remainder Theorem (Return a number x which satisfies x = a mod m & x = b mod n)
// All the values has to be less than 2^30
// O(log(m + n))
ll euclid(ll x, ll y, ll &a, ll &b){
	if(y){
		ll d = euclid(y, x % y, b, a);
		return b -= x / y * a, d;
	}
	return a = 1, b = 0, x;
}
ll crt_coprime(ll a, ll m, ll b, ll n){
	ll x, y; euclid(m, n, x, y);
	ll res = a * (y + m) % m * n + b * (x + n) % n * m;
	if(res >= m * n) res -= m * n;
	return res;
}
ll crt(ll a, ll m, ll b, ll n){
	ll d = gcd(m, n);
	if(((b -= a) %= n) < 0) b += n;
	if(b % d) return -1; // No solution
	return d * crt_coprime(0LL, m/d, b/d, n/d) + a;
}

// 156485479_1_10
// Lehman Factorization / return a prime divisor of x
// x has to be equal or less than 10^14
// O(N^1/3)
ll primefactor(ll x){
	assert(x > 1);
	if(x <= 21){
		for(ll p = 2; p <= sqrt(x); ++ p) if(x % p == 0) return p;
		return x;
	}
	for(ll p = 2; p <= cbrt(x); ++ p) if(x % p == 0) return p;
	for(ll k = 1; k <= cbrt(x); ++ k){
		double t = 2 * sqrt(k * x);
		for(ll a = ceil(t); a <= floor(t + cbrt(sqrt(x)) / 4 / sqrt(k)); ++ a){
			ll b = a * a - 4 * k * x, s = sqrt(b);
			if(b == s * s) return gcd(a + s, x);
		}
	}
	return x;
}

// 156485479_1_11
// Mobius Function
// O(n)
void linearsieve(int n, vi &lpf, vi &prime){
	lpf.resize(n + 1);
	prime.reserve(n + 1);
	for(int i = 2; i <= n; i ++){
		if(!lpf[i]){
			lpf[i] = i;
			prime.push_back(i);
		}
		for(int j = 0; j < prime.size() && prime[j] <= lpf[i] && i * prime[j] <= n; j ++){
			lpf[i * prime[j]] = prime[j];
		}
	}
}
void process_mobius(int n, vi &mu){
	vi lpf, prime;
	linearsieve(n, lpf, prime);
	mu.resize(n + 1);
	mu[1] = 1;
	for(int i = 2; i <= n; ++ i){
		mu[i] = (i / lpf[i] % lpf[i] ? -mu[i / lpf[i]] : 0);
	}
}

// 156485479_2_1
// Linear Recurrence Relation Solver / Berlekamp - Massey Algorithm
// O(N^2 log n) / O(N^2)
struct recurrence{
	int N;
	vl init, coef;
	ll mod;
	recurrence(vl init, vl coef, ll mod):
		N(coef.size()), init(init), coef(coef), mod(mod){
	}
	// Berlekamp Massey Algorithm
	recurrence(vl s, ll mod): mod(mod){
		int n = s.size();
		N = 0;
		vl B(n), T;
		coef.resize(n);
		coef[0] = B[0] = 1;
		ll b = 1;
		for(int i = 0, m = 0; i < n; i ++){
			m ++;
			ll d = s[i] % mod;
			for(int j = 1; j <= N; j ++){
				d = (d + coef[j] * s[i - j]) % mod;
			}
			if(!d) continue;
			T = coef;
			ll c = d * modexp(b, mod - 2, mod) % mod;
			for(int j = m; j < n; j ++){
				coef[j] = (coef[j] - c * B[j - m]) % mod;
			}
			if(2 * N > i) continue;
			N = i + 1 - N, B = T, b = d, m = 0;
		}
		coef.resize(N + 1), coef.erase(coef.begin());
		for(auto &x: coef){
			x = (mod - x) % mod;
		}
		reverse(all(coef));
		init.resize(N);
		for(int i = 0; i < N; i ++){
			init[i] = s[i] % mod;
		}
	}
	// O(N^2 log n)
	ll operator[](ll n) const{
		auto combine = [&](vl a, vl b){
			vl res(2 * N + 1);
			for(int i = 0; i <= N; i ++){
				for(int j = 0; j <= N; j ++){
					res[i + j] = (res[i + j] + a[i] * b[j]) % mod;
				}
			}
			for(int i = 2 * N; i > N; i --){
				for(int j = 0; j < N; j ++){
					res[i - 1 - j] = (res[i - 1 - j] + res[i] * coef[N - 1 - j]) % mod;
				}
			}
			res.resize(N + 1);
			return res;
		};
		vl pol(N + 1), e(pol);
		pol[0] = e[1] = 1;
		for(n ++; n; n /= 2){
			if(n % 2) pol = combine(pol, e);
			e = combine(e, e);
		}
		ll res = 0;
		for(int i = 0; i < N; i ++){
			res = (res + pol[i + 1] * init[i]) % mod;
		}
		return res;
	}
};

// 156485479_2_2_1
// Find a solution of the system of linear equations. Return -1 if no sol, rank otherwise.
// O(n^2 m)
const double eps = 1e-12;
int solve_linear_equations(const vector<vd> &AA, vd &x, const vd &bb){
	auto A = AA;
	auto b = bb;
	int n = A.size(), m = A[0].size(), rank = 0, br, bc;
	vi col(m);
	iota(col.begin(), col.end(), 0);
	for(int i = 0; i < n; i ++){
		double v, bv = 0;
		for(int r = i; r < n; r ++){
			for(int c = i; c < m; c ++){
				if((v = fabs(A[r][c])) > bv){
					br = r, bc = c, bv = v;
				}
			}
		}
		if(bv <= eps){
			for(int j = i; j < n; j ++){
				if(fabs(b[j]) > eps){
					return -1;
				}
			}
			break;
		}
		swap(A[i], A[br]), swap(b[i], b[br]), swap(col[i], col[bc]);
		for(int j = 0; j < n; j ++){
			swap(A[j][i], A[j][bc]);
		}
		bv = 1 / A[i][i];
		for(int j = i + 1; j < n; j ++){
			double fac = A[j][i] * bv;
			b[j] -= fac * b[i];
			for(int k = i + 1; k < m; k ++){
				A[j][k] -= fac * A[i][k];
			}
		}
		rank ++;
	}
	x.resize(m);
	for(int i = rank; i --; ){
		b[i] /= A[i][i];
		x[col[i]] = b[i];
		for(int j = 0; j < i; j ++){
			b[j] -= A[j][i] * b[i];
		}
	}
	return rank;
}

// 156485479_2_2_2
// Find a solution of the system of linear equations. Return -1 if no sol, rank otherwise.
// O(n^2 m)
int solve_linear_equations(const vector<vl> &AA, vl &x, const vl &bb, ll mod){
	auto A = AA;
	auto b = bb;
	int n = A.size(), m = A[0].size(), rank = 0, br, bc;
	vi col(m);
	iota(col.begin(), col.end(), 0);
	for(auto &x: A){
		for(auto &y: x){
			y %= mod;
		}
	}
	for(int i = 0; i < n; i ++){
		long long v, bv = 0;
		for(int r = i; r < n; r ++){
			for(int c = i; c < m; c ++){
				if((v = abs(A[r][c])) > bv){
					br = r, bc = c, bv = v;
				}
			}
		}
		if(!bv){
			for(int j = i; j < n; j ++){
				if(abs(b[j])){
					return -1;
				}
			}
			break;
		}
		swap(A[i], A[br]), swap(b[i], b[br]), swap(col[i], col[bc]);
		for(int j = 0; j < n; j ++){
			swap(A[j][i], A[j][bc]);
		}
		bv = modinv(A[i][i], mod);
		for(int j = i + 1; j < n; j ++){
			ll fac = A[j][i] * bv % mod;
			b[j] = (b[j] - fac * b[i] % mod + mod) % mod;
			for(int k = i + 1; k < m; k ++){
				A[j][k] = (A[j][k] - fac * A[i][k] % mod + mod) % mod;
			}
		}
		rank ++;
	}
	x.resize(m);
	for(int i = rank; i --; ){
		b[i] = b[i] * modinv(A[i][i], mod) % mod;
		x[col[i]] = b[i];
		for(int j = 0; j < i; j ++){
			b[j] = (b[j] - A[j][i] * b[i] % mod + mod) % mod;
		}
	}
	return rank;
}

// 156485479_2_2_3
// Find a solution of the system of linear equations. Return -1 if no sol, rank otherwise.
// O(n^2 m)
typedef bitset<1000> bs;
int solve_linear_equations(const vector<bs> &AA, bs& x, const vi &bb, int m){
	vector<bs> A(AA);
	vi b(bb);
	int n = sz(A), rank = 0, br;
	vi col(m);
	iota(all(col), 0);
	for(int i = 0; i < n; ++ i){
		for(br = i; br < n; ++ br) if(A[br].any()) break;
		if (br == n){
			for(int j = i; j < n; ++ j) if(b[j]) return -1;
			break;
		}
		int bc = (int)A[br]._Find_next(i-1);
		swap(A[i], A[br]);
		swap(b[i], b[br]);
		swap(col[i], col[bc]);
		for(int j = 0; j < n; ++ j) if(A[j][i] != A[j][bc]) A[j].flip(i); A[j].flip(bc);
		for(int j = i + 1; j < n; ++ j) if(A[j][i]) b[j] ^= b[i], A[j] ^= A[i];
		++ rank;
	}
	x = bs();
	for (int i = rank; -- i;){
		if (!b[i]) continue;
		x[col[i]] = 1;
		for(int j = 0; j < i; ++ j) b[j] ^= A[j][i];
	}
	return rank;
}

// 156485479_2_3_1
// Matrix for Z_p
struct matrix: vector<vl>{
	int N, M;
	const ll mod;
	matrix(int N, int M, ll mod, int flag = 0): N(N), M(M), mod(mod){
		resize(N, vl(M));
		if(flag){
			int temp = min(N, M);
			for(int i = 0; i < temp; ++ i) (*this)[i][i] = 1;
		}
	}
	matrix(const vector<vl> &arr, ll mod): N(sz(arr)), M(sz(arr[0])), mod(mod){
		resize(N);
		for(int i = 0; i < N; ++ i) (*this)[i] = arr[i];
	}
	matrix operator=(const matrix &other){
		N = other.N, M = other.M;
		resize(N);
		for(int i = 0; i < N; ++ i) (*this)[i] = other[i];
		return *this;
	}
	matrix operator+(const matrix &other) const{
		matrix res(N, M, mod);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) res[i][j] = ((*this)[i][j] + other[i][j]) % mod;
		return res;
	}
	matrix operator+=(const matrix &other){
		*this = *this + other;
		return *this;
	}
	matrix operator*(const matrix &other) const{
		assert(M == other.N);
		int L = other.M;
		matrix res(N, M, mod);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < L; ++ j) for(int k = 0; k < M; ++ k) (res[i][j] += (*this)[i][k] * other[k][j]) %= mod;
		return res;
	}
	matrix operator*=(const matrix &other){
		*this = *this * other;
		return *this;
	}
	matrix operator^(ll e) const{
		assert(N == M);
		matrix res(N, N, mod, 1), b(*this);
		while(e){
			if(e & 1) res *= b;
			b *= b;
			e >>= 1;
		}
		return res;
	}
	ll det(){
		assert(N == M);
		vector<vl> temp = *this;
		ll res = 1;
		for(int i = 0; i < N; ++ i){
			for(int j = i + 1; j < N; ++ j){
				while(temp[j][i]){
					ll t = temp[i][i] / temp[j][i];
					if(t) for(int k = i; i < N; ++ k) temp[i][k] = (temp[i][k] - temp[j][k] * t) % mod;
					std::swap(temp[i], temp[j]);
					res *= -1;
				}
			}
			res = res * temp[i][i] % mod;
			if(!res) return 0;
		}
		return (res + mod) % mod;
	}
};

// 156485479_2_3_2
// Matrix for general ring
template<class T>
struct matrix: vector<vector<T>>{
	int N, M;
	T aid, mid; // multiplicative identity
	matrix(int N, int M, const T &aid, const T &mid, int flag):
		N(N), M(M), aid(aid), mid(mid){
		this->resize(N, vector<T>(M, aid));
		if(flag) for(int i = 0; i < min(N, M); ++ i) (*this)[i][i] = mid;
	}
	matrix(const vector<vector<T>> &arr, const T &aid, const T &mid): N(sz(arr)), M(sz(arr[0])), aid(aid), mid(mid){
		this->resize(N, vector<T>(M, aid));
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) (*this)[i][j] = arr[i][j];
	}
	matrix operator=(const matrix &other){
		N = other.N, M = other.M;
		this->resize(N);
		for(int i = 0; i < N; ++ i) (*this)[i] = other[i];
		return *this;
	}
	matrix operator+(const matrix &other) const{
		matrix res(N, M, aid, mid, 0);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) res[i][j] = (*this)[i][j] + other[i][j];
		return res;
	}
	matrix operator+=(const matrix &other){
		*this = *this + other;
		return *this;
	}
	matrix operator*(const matrix &other) const{
		assert(M == other.N);
		int L = other.M;
		matrix res(N, M, aid, mid, 0);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < L; ++ j) for(int k = 0; k < M; ++ k) res[i][j] = res[i][j] + (*this)[i][k] * other[k][j];
		return res;
	}
	matrix operator*=(const matrix &other){
		*this = *this * other;
		return *this;
	}
	matrix operator^(ll e) const{
		assert(N == M);
		matrix res(N, N, aid, mid, 1), b(*this);
		while(e){
			if(e & 1) res *= b;
			b *= b;
			e >>= 1;
		}
		return res;
	}
};

// 156485479_2_4_1_1_1
// Fast Fourier Transformation.
// Size must be a power of two.
// O(n log n)
typedef complex<double> cd;
const double PI = acos(-1);
void fft(vector<cd> &f, bool invert){
	int n = sz(f);
	for(int i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(f[i], f[j]);
	}
	for(int len = 2; len <= n; len <<= 1){
		double theta = 2 * PI / len * (invert ? -1 : 1);
		cd w(cos(theta), sin(theta));
		for(int i = 0; i < n; i += len){
			cd wj(1);
			for(int j = 0; j < len / 2; ++ j, wj *= w){
				cd u = f[i + j], v = wj * f[i + j + len / 2];
				f[i + j] = u + v, f[i + j + len / 2] = u - v;
			}
		}
	}
	if(invert) for(auto &c: f) c /= n;
}
vl polymul(const vl &a, const vl &b){
	vector<cd> f(all(a)), g(all(b));
	int n = 1;
	while(n < a.size() + b.size()) n <<= 1;
	f.resize(n), g.resize(n);
	fft(f, false), fft(g, false);
	for(int i = 0; i < n; i ++) f[i] *= g[i];
	fft(f, true);
	vl res(n);
	for(int i = 0; i < n; ++ i) res[i] = round(f[i].real());
	while(!res.empty() && !res.back()) res.pop_back();
	return res;
}

// 156485479_2_4_1_1_2
// Number Theoric Transformation. Use (998244353, 15311432, 1 << 23) or (7340033, 5, 1 << 20)
// Size must be a power of two
// O(n log n)
ll modexp(ll b, ll e, const ll &mod){
	ll res = 1;
	for(; e; b = b * b % mod, e /= 2) if(e & 1) res = res * b % mod;
	return res;
}
ll modinv(ll b, const ll &mod){
	return modexp(b, mod - 2, mod);
}
const ll mod = 998244353, root = 15311432, root_pw = 1 << 23, root_1 = modinv(root, mod);
vl ntt(const vl &arr, bool invert){
    int n = sz(arr);
    vl a{arr};
    for(int i = 1, j = 0; i < n; ++ i){
        int bit = n >> 1;
        for(; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if(i < j) swap(a[i], a[j]);
    }
    for(int len = 2; len <= n; len <<= 1){
        ll wlen = invert ? root_1 : root;
        for(int i = len; i < root_pw; i <<= 1) wlen = wlen * wlen % mod;
        for(int i = 0; i < n; i += len){
            ll w = 1;
            for(int j = 0; j < len / 2; ++ j){
                ll u = a[i + j], v = a[i + j + len / 2] * w % mod;
                a[i + j] = u + v < mod ? u + v : u + v - mod;
                a[i + j + len / 2] = u - v >= 0 ? u - v : u - v + mod;
                w = w * wlen % mod;
            }
        }
    }
    if(invert){
        ll n_1 = modinv(n, mod);
        for(auto &x: a) x = x * n_1 % mod;
    }
    return a;
}

// 156485479_2_4_1_2
// Bitwise XOR Transformation ( Fast Walsh Hadamard Transformation, FWHT ).
// Size must be a power of two.
// Transformation   1  1     Inversion   1  1     
//     Matrix       1 -1      Matrix     1 -1   TIMES  1/2
// O(n log n)
template<class T>
vector<T> xort(const vector<T> &P, bool inverse){
	vector<T> p(P);
	int n = sz(p);
	for(int len = 1; 2 * len <= n; len <<= 1){
		for(int i = 0; i < n; i += 2 * len){
			for(int j = 0; j < len; ++ j){
				T u = p[i + j], v = p[i + j + len];
				p[i + j] = u + v, p[i + j + len] = u - v;
			}
		}
	}
	if(inverse){
		for(int i = 0; i < n; ++ i) p[i] /= n;
	}
	return p;
}

// 156485479_2_4_1_3
// Bitwise AND Transformation.
// Size must be a power of two.
// Transformation   0  1     Inversion   -1  1
//     Matrix       1  1      Matrix      1  0
// O(n log n)
template<class T>
vector<T> andt(const vector<T> &P, bool inverse){
	vector<T> p(P);
	int n = sz(p);
	for(int len = 1; 2 * len <= n; len <<= 1){
		for(int i = 0; i < n; i += 2 * len){
			for(int j = 0; j < len; ++ j){
				T u = p[i + j], v = p[i + j + len];
				if(!inverse) p[i + j] = v, p[i + j + len] = u + v;
				else p[i + j] = -u + v, p[i + j + len] = u;
			}
		}
	}
	return p;
}

// 156485479_2_4_1_4
// Bitwise OR Transformation.
// Size must be a power of two
// Transformation   1  1     Inversion    0  1
//     Matrix       1  0      Matrix      1 -1
// O(n log n)
template<class T>
vector<T> ort(const vector<T> &P, bool inverse){
	vector<T> p(P);
	int n = sz(p);
	for(int len = 1; 2 * len <= n; len <<= 1){
		for(int i = 0; i < n; i += 2 * len){
			for(int j = 0; j < len; ++ j){
				T u = p[i + j], v = p[i + j + len];
				if(!inverse) p[i + j] = u + v, p[i + j + len] = u;
				else p[i + j] = v, p[i + j + len] = u - v;
			}
		}
	}
	return p;
}

// 156485479_2_4_2_1_1
// Polynomial Interpolation
// O(n ^ 2)
vd interpolate(vd x, vd y){
	int n = x.size();
	vd res(n), temp(n);
	for(int k = 0; k < n; ++ k) for(int i = k + 1; i < n; ++ i){
		y[i] = (y[i] - y[k]) / (x[i] - x[k]);
	}
	double last = 0; temp[0] = 1;
	for(int k = 0; k < n; ++ k) for(int i = 0; i < n; ++ i){
		res[i] += y[k] * temp[i];
		swap(last, temp[i]);
		temp[i] -= last * x[k];
	}
	return res;
}

// 156485479_2_4_2_1_2
// Polynomial Interpolation
// O(n ^ 2)
vl interpolate(vl x, vl y, ll mod){
	int n = sz(x);
	vl res(n), temp(n);
	for(int k = 0; k < n; ++ k){
		for(int i = k + 1; i < n; ++ i){
			y[i] = (y[i] - y[k]) * modinv(x[i] - x[k], mod) % mod;
		}
	}
	ll last = 0; temp[0] = 1;
	for(int k = 0; k < n; ++ k){
		for(int i = 0; i < n; ++ i) {
			res[i] = (res[i] + y[k] * temp[i]) % mod;
			swap(last, temp[i]);
			temp[i] = (temp[i] - last * x[k] % mod + mod) % mod;
		}
	}
	return res;
}

// 156485479_2_4_2_2
// Polynomial Interpolation
// O(n log n)
// (INCOMPLETE!)

// 156485479_2_5
// Kadane
// O(N)
template<class T>
T kadane(const vector<T> &arr){
	int n = sz(arr);
	T lm = 0, gm = 0;
	for(int i = 0; i < n; ++ i){
		lm = max(arr[i], arr[i] + lm);
		gm = max(gm, lm);
	}
	return gm;
}

// 156485479_2_6_1_1
// Sorted Line Container
// O(log N) per query, amortized O(1) for everything else
struct line{
	ll d, k, p;
	int ind;
	ll eval(ll x){ return d * x + k; }
};
template<bool GET_MAX = true>
struct sorted_line_container: deque<line>{
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	const ll inf = LLONG_MAX;
	ll div(ll a, ll b){ return a / b - ((a ^ b) < 0 && a % b); }
	bool isect_front(iterator x, iterator y){
		if(y == this->end()){ x->p = inf; return false; }
		else{ x->p = div(y->k - x->k, x->d - y->d); return x->p >= y->p; }
	}
	bool isect_back(reverse_iterator x, reverse_iterator y){
		if(x == this->rend()) return false;
		else{ x->p = div(y->k - x->k, x->d - y->d); return x->p >= y->p; }
	}
	void push(line L){
		if(!GET_MAX) L.d = -L.d, L.k = -L.k;
		if(empty() || L.d < front().d){
			L.p = 0, push_front(L), isect_front(begin(), ++ begin());
			while(size() >= 2 && isect_front(begin(), ++ begin())) erase(++ begin());
		}
		else if(L.d > back().d){
			L.p = inf, push_back(L); isect_back(++ rbegin(), rbegin());
			while(size() >= 2 && isect_back(++ ++ rbegin(), ++ rbegin())) erase(-- -- end()), isect_back(++ rbegin(), rbegin());
		}
		else assert(false);
	}
	ll dec_query(ll x){
		while(size() >= 2 && rbegin()->eval(x) <= (++ rbegin())->eval(x)) pop_back(); rbegin()->p = inf;
		return rbegin()->eval(x) * (GET_MAX ? 1 : -1);
	}
	ll inc_query(ll x){
		while(size() >= 2 && begin()->eval(x) <= (++ begin())->eval(x)) pop_front();
		return begin()->eval(x) * (GET_MAX ? 1 : -1);
	}
	ll query(ll x){
		if(size() == 1) return begin()->eval(x) * (GET_MAX ? 1 : -1);
		int low = 0, high = int(size()) - 1;
		if(begin()->eval(x) >= (++ begin())->eval(x)) return begin()->eval(x) * (GET_MAX ? 1 : -1);
		while(high - low > 1){
			int mid = low + high >> 1;
			(*this)[mid].eval(x) < (*this)[mid + 1].eval(x) ? low = mid : high = mid;
		}
		return (*this)[low + 1].eval(x) * (GET_MAX ? 1 : -1);
	}
};

// 156485479_2_6_1_2
// Line Container / Add lines of form d*x + k and query max at pos x
// O(log N) per query
struct line{
	mutable ll d, k, p;
	bool operator<(const line &otr) const{ return d < otr.d; }
	bool operator<(ll x) const{ return p < x;}
};
template<bool GET_MAX = true>
struct line_container: multiset<line, less<>>{
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	const ll inf = LLONG_MAX;
	ll div(ll a, ll b){ return a / b - ((a ^ b) < 0 && a % b); }
	bool isect(iterator x, iterator y){
		if(y == this->end()){ x->p = inf; return false; }
		if(x->d == y->d) x->p = x->k > y->k ? inf : -inf;
		else x->p = div(y->k - x->k, x->d - y->d);
		return x->p >= y->p;
	}
	void push(line L){
		if(!GET_MAX) L.d = -L.d, L.k = -L.k;
		L.p = 0;
		auto z = this->insert(L), y = z ++, x = y;
		while(isect(y, z)) z = this->erase(z);
		if(x != this->begin() && isect(-- x, y)) isect(x, y = this->erase(y));
		while((y = x) != this->begin() && (-- x)->p >= y->p) isect(x, this->erase(y));
	}
	ll query(ll x){
		assert(!this->empty());
		auto l = *this->lower_bound(x);
		return (l.d * x + l.k) * (GET_MAX ? 1 : -1);
	}
};

// 156485479_2_6_1_3
// Li Chao Tree
// O(log N) per update and query
struct line{
	ll d, k;
	line(ll d = 0, ll k = -(ll)9e18): d(d), k(k){ }
	ll eval(ll x){ return d * x + k; }
	bool majorize(line X, ll L, ll R){ return eval(L) >= X.eval(L) && eval(R) >= X.eval(R); }
};
template<bool GET_MAX = true>
struct lichao{
	lichao *l = NULL, *r = NULL;
	line S;
	lichao(): S(line()){ }
	void rm(){
		if(l) l->rm();
		if(r) r->rm();
		delete this;
	}
	void mc(int i){
		if(i){ if(!r) r = new lichao(); }
		else{ if(!l) l = new lichao(); }
	}
	ll pq(ll X, ll L, ll R){
		ll ans = S.eval(X), M = L + R >> 1;
		if(X < M) return max(ans, l ? l->pq(X, L, M) : -(ll)9e18);
		else return max(ans, r ? r->pq(X, M, R) : -(ll)9e18);
	}
	ll query(ll X, ll L, ll R){
		return pq(X, L, R) * (GET_MAX ? 1 : -1);
	}
	void pp(line X, ll L, ll R){
		if(X.majorize(S, L, R)) swap(X, S);
		if(S.majorize(X, L, R)) return;
		if(S.eval(L) < X.eval(L)) swap(X, S);
		ll M = L + R >> 1;
		if(X.eval(M) > S.eval(M)) swap(X, S), mc(0), l->pp(X, L, M);
		else mc(1), r->pp(X, M, R);
	}
	void push(line X, ll L, ll R){
		if(!GET_MAX) X.d = -X.d, X.k = -X.k;
		pp(X, L, R);
	}
};

// 156485479_2_6_2
// Divide and Conquer DP Optimization
// Recurrence relation of form dp_next[i] = min{j in [0, i)} (dp[j] + C[j][i])
// Must satisfy opt[j] <= opt[j + 1]
// Special case: for all a<=b<=c<=d, C[a][c] + C[b][d] <= C[a][d] + C[b][d] ( C is a Monge array )
// O(N log N)
template<class T>
void DCDP(vector<T> &dp, vector<T> &dp_next, const vector<vector<T>> &C, int low, int high, int optl, int optr){
	if(low >= high) return;
	int mid = low + high >> 1;
	pair<T, int> res{numeric_limits<T>::max(), -1};
	for(int i = optl; i < min(mid, optr); ++ i) res = min(res, {dp[i] + C[i][mid], i});
	dp_next[mid] = res.first;
	DCDP(dp, dp_next, C, low, mid, optl, res.second + 1);
	DCDP(dp, dp_next, C, mid + 1, high, res.second, optr);
}

// 156485479_2_6_3
// Knuth DP Optimization
// Recurrence relation of form dp[i][j] = min{k in [i, j)} (dp[i][k] + dp[k][j] + C[i][j])
// Must satisfy C[a][c] + C[b][d] <= C[a][d] + C[b][d] (C is a monge array) and C[a][d] >= C[b][c] for all a<=b<=c<=d
// It can be proved that opt[i][j - 1] <= opt[i][j] <= opt[i + 1][j]
// Fill the dp table in increasing order of j - i.
// O(N^2)

// 156485479_2_6_4
// Lagrange DP Optimization ( Aliens Trick, Wqs Binary Search )
// Recurrence relation of form dp[i][j] = min{k in [0, j)} (dp[i - 1][k] + C[k + 1][j])
// dp[x][N] must be convex / concave
// Special case: for all a<=b<=c<=d, C[a][c] + C[b][d] <= C[a][d] + C[b][d] ( C is a Monge array )
// f(const ll &lambda, vi &previous, vi &count) returns the reduced DP value
// WARNING: the cost function for f() should be doubled
// O(log(high - low)) applications of f()
template<class Pred>
ll custom_binary_search(ll low, ll high, const ll &step, Pred p, bool is_left = true){
	assert(low < high && (high - low) % step == 0);
	const ll rem = low % step;
	if(is_left){
		while(high - low > step){
			ll mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? low = mid : high = mid;
		}
		return low;
	}
	else{
		while(high - low > step){
			ll mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? high = mid : low = mid;
		}
		return high;
	}
}
template<class DP, bool GET_MAX = true>
pair<ll, vi> LagrangeDP(int n, DP f, ll k, ll low, ll high){
	ll resp, resq;
	vi prevp(n + 1), cntp(n + 1), prevq(n + 1), cntq(n + 1);
	auto pred = [&](ll lambda){
		swap(resp, resq), swap(prevp, prevq), swap(cntp, cntq);
		resp = f(lambda, prevp, cntp);
		return GET_MAX ? cntp.back() <= k : cntp.back() >= k;
	};
	ll lambda = custom_binary_search(2 * low - 1, 2 * high + 1, 2, pred);
	pred(lambda + 2), pred(lambda);
	if(cntp.back() == k){
		vi path{n};
		for(int u = n; u; ) path.push_back(u = prevp[u]);
		return {resp - lambda * k >> 1, path};
	}
	else{
		resp = resp - lambda * cntp.back() >> 1, resq = resq - (lambda + 2) * cntq.back() >> 1;
		ll res = resp + (resq - resp) / (cntq.back() - cntp.back()) * (k - cntp.back());
		if(!GET_MAX) swap(prevp, prevq), swap(cntp, cntq);
		int i = n, j = n, d = k - cntp.back();
		while(1){
			if(prevp[i] <= prevq[j]){
				while(prevp[i] <= prevq[j] && cntq[j] - cntp[i] > d) j = prevq[j];
				if(prevp[i] <= prevq[j] && cntq[j] - cntp[i] == d) break;
			}
			else i = prevp[i], j = prevq[j];
		}
		vi path{n};
		for(int u = n; u != i; ) path.push_back(u = prevp[u]);
		path.push_back(prevq[j]);
		for(int u = prevq[j]; u; ) path.push_back(u = prevq[u]);
		return {res, path};
	}
}

// 156485479_2_7
// Binary Search
// O(log(high - low)) applications of p
template<class Pred>
ll custom_binary_search(ll low, ll high, Pred p, bool is_left = true){
	assert(low < high);
	if(is_left){
		while(high - low > 1){
			ll mid = low + (high - low >> 1);
			p(mid) ? low = mid : high = mid;
		}
		return low;
	}
	else{
		while(high - low > 1){
			ll mid = low + (high - low >> 1);
			p(mid) ? high = mid : low = mid;
		}
		return high;
	}
}

// Binary search for numbers with the same remainder mod step
template<class Pred>
ll custom_binary_search(ll low, ll high, const ll &step, Pred p, bool is_left = true){
	assert(low < high && (high - low) % step == 0);
	const ll rem = low % step;
	if(is_left){
		while(high - low > step){
			ll mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? low = mid : high = mid;
		}
		return low;
	}
	else{
		while(high - low > step){
			ll mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? high = mid : low = mid;
		}
		return high;
	}
}

// 156485479_3_1
// Sparse Table
// The binary operator must be idempotent and associative
// O(N log N) preprocessing, O(1) per query
template<class T, class BO = function<T(T, T)>>
struct sparse_table: vector<vector<T>>{
	int N;
	BO bin_op;
	sparse_table(const vector<T> &arr, BO bin_op = [](T x, T y){return min(x, y);}): N(arr.size()), bin_op(bin_op){
		int t = 1, d = 1;
		while(t < N) t *= 2, ++ d;
		this->assign(d, arr);
		for(int i = 0; i < d - 1; ++ i) for(int j = 0; j < N; ++ j){
			(*this)[i + 1][j] = bin_op((*this)[i][j], (*this)[i][min(N - 1, j + (1 << i))]);
		}
	}
	sparse_table(){ }
	T query(int l, int r){
		int d = 31 - __builtin_clz(r - l);
		return bin_op((*this)[d][l], (*this)[d][r - (1 << d)]);
	}
	sparse_table &operator=(const sparse_table &otr){
		N = otr.N, bin_op = otr.bin_op; this->clear();
		for(auto &t: otr) this->push_back(t);
		return *this;
	}
};

// 156485479_3_2_1
// Iterative Segment Tree
// O(N) processing, O(log N) per query
template<class T, class BO>
struct segment: vector<T>{
	int N;
	BO bin_op;
	const T id;
	segment(const vector<T> &arr, BO bin_op, T id): N(sz(arr)), bin_op(bin_op), id(id){
		this->resize(N << 1, id);
		for(int i = 0; i < N; ++ i) (*this)[i + N] = arr[i];
		for(int i = N - 1; i > 0; -- i) (*this)[i] = bin_op((*this)[i << 1], (*this)[i << 1 | 1]);
	}
	void set(int p, T val){
		for((*this)[p += N] = val; p > 1; p >>= 1) (*this)[p >> 1] = bin_op((*this)[p], (*this)[p ^ 1]);
	}
	T query(int l, int r){
		if(l >= r) return id;
		T resl = id, resr = id;
		for(l += N, r += N; l < r; l >>= 1, r >>= 1){
			if(l & 1) resl = bin_op(resl, (*this)[l ++]);
			if(r & 1) resr = bin_op((*this)[-- r], resr);
		}
		return bin_op(resl, resr);
	}
};

// 156485479_3_2_2
// Iterative Segment Tree with Reversed Operation ( Commutative Operation Only )
// O(N) Preprocessing, O(1) per query
template<class T, class BO>
struct reversed_segment: vector<T>{
	int N;
	BO bin_op;
	T id;
	reversed_segment(const vector<T> &arr, BO bin_op, T id): N(sz(arr)), bin_op(bin_op), id(id){
		this->resize(N << 1, id);
		for(int i = 0; i < N; ++ i) (*this)[i + N] = arr[i];
	}
	void update(int l, int r, T val){
		for(l += N, r += N; l < r; l >>= 1, r >>= 1){
			if(l & 1) (*this)[l ++] = bin_op((*this)[l], val);
			if(r & 1) (*this)[r] = bin_op((*this)[-- r], val);
		}
	}
	T query(int p){
		T res = id;
		for(p += N; p > 0; p >>= 1) res = bin_op(res, (*this)[p]);
		return res;
	}
	void push(){
		for(int i = 1; i < N; ++ i){
			(*this)[i << 1] = bin_op((*this)[i << 1], (*this)[i]);
			(*this)[i << 1 | 1] = bin_op((*this)[i << 1 | 1], (*this)[i]);
			(*this)[i] = id;
		}
	}
};

// 156485479_3_2_3
// Iterative Segment Tree Supporting Lazy Propagation
// INCOMPLETE


// 156485479_3_2_4
// Simple Recursive Segment Tree
// O(N) preprocessing, O(log N) per query
template<class T, class BO>
struct recursive_segment{
	int N;
	vector<T> arr;
	BO bin_op;
	T id;
	recursive_segment(const vector<T> &arr, BO bin_op, T id): N(sz(arr)), bin_op(bin_op), id(id){
		this->arr.resize(N << 2, id);
		build(arr, 1, 0, N);
	}
	void build(const vector<T> &arr, int u, int left, int right){
		if(left + 1 == right) this->arr[u] = arr[left];
		else{
			int mid = left + right >> 1;
			build(arr, u << 1, left, mid);
			build(arr, u << 1 ^ 1, mid, right);
			this->arr[u] = bin_op(this->arr[u << 1], this->arr[u << 1 ^ 1]);
		}
	}
	T pq(int u, int left, int right, int ql, int qr){
		if(qr <= left || right <= ql) return id;
		if(ql == left && qr == right) return arr[u];
		int mid = left + right >> 1;
		return bin_op(pq(u << 1, left, mid, ql, qr), pq(u << 1 ^ 1, mid, right, ql, qr));
	}
	T query(int ql, int qr){
		return pq(1, 0, N, ql, qr);
	}
	void pu(int u, int left, int right, int ind, T val){
		if(left + 1 == right) arr[u] = val;
		else{
			int mid = left + right >> 1;
			if(ind < mid) pu(u << 1, left, mid, ind, val);
			else pu(u << 1 ^ 1, mid, right, ind, val);
			arr[u] = bin_op(arr[u << 1], arr[u << 1 ^ 1]);
		}
	}
	void update(int ind, T val){
		pu(1, 0, N, ind, val);
	}
	// Below assumes T is an ordered field and node stores positive values
	template<class IO>
	int plb(int u, int left, int right, T val, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(arr[u << 1] < val) return plb(u << 1 ^ 1, mid, right, inv_op(val, arr[u << 1]), inv_op);
		else return plb(u << 1, left, mid, val, inv_op);
	}
	template<class IO>
	int lower_bound(T val, IO inv_op){ // min i such that query[0, i) >= val
		if(arr[1] < val) return N + 1;
		else return plb(1, 0, N, val, inv_op);
	}
	template<class IO>
	int lower_bound(int i, T val, IO inv_op){
		return lower_bound(bin_op(val, query(0, min(i, N))), inv_op);
	}
	template<class IO>
	int pub(int u, int left, int right, T val, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(val < arr[u << 1]) return pub(u << 1, left, mid, val, inv_op);
		else return pub(u << 1 ^ 1, mid, right, inv_op(val, arr[u << 1]), inv_op);
	}
	template<class IO>
	int upper_bound(T val, IO inv_op){ // min i such that query[0, i) > val
		if(val < arr[1]) return pub(1, 0, N, val, inv_op);
		else return N + 1;
	}
	template<class IO>
	int upper_bound(int i, T val, IO inv_op){
		return upper_bound(bin_op(val, query(0, min(i, N))), inv_op);
	}
};

// 156485479_3_2_5
// Implement 2D Segment Tree Here
// 

// 156485479_3_2_6
// Lazy Dynamic Segment Tree
// O(1) or O(N) preprocessing, O(log L) or O(log N) per query
template<class T, class BO1, class BO2, class BO3>
struct lazy_segment{
	lazy_segment *l = 0, *r = 0;
	int low, high;
	BO1 lop;           // Lazy op(L, L -> L)
	BO2 qop;           // Query op(Q, Q -> Q)
	BO3 aop;           // Apply op(Q, L, len -> Q)
	vector<T> &id;     // Lazy id(L), Query id(Q), Disable constant(Q)
	T lset, lazy, val;
	lazy_segment(int low, int high, BO1 lop, BO2 qop, BO3 aop, vector<T> &id): low(low), high(high), lop(lop), qop(qop), aop(aop), id(id){
		lazy = id[0], val = id[1], lset = id[2];
	}
	lazy_segment(const vector<T> &arr, int low, int high, BO1 lop, BO2 qop, BO3 aop, vector<T> &id)
	: low(low), high(high), lop(lop), qop(qop), aop(aop), id(id){
		lazy = id[0], lset = id[2];
		if(high - low > 1){
			int mid = low + (high - low) / 2;
			l = new lazy_segment(arr, low, mid, lop, qop, aop, id);
			r = new lazy_segment(arr, mid, high, lop, qop, aop, id);
			val = qop(l->val, r->val);
		}
		else val = arr[low];
	}
	void push(){
		if(!l){
			int mid = low + (high - low) / 2;
			l = new lazy_segment(low, mid, lop, qop, aop, id);
			r = new lazy_segment(mid, high, lop, qop, aop, id);
		}
		if(lset != id[2]){
			l->set(low, high, lset);
			r->set(low, high, lset);
			lset = id[2];
		}
		else if(lazy != id[0]){
			l->update(low, high, lazy);
			r->update(low, high, lazy);
			lazy = id[0];
		}
	}
	void set(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			lset = x;
			lazy = id[0];
			val = aop(id[1], x, high - low);
		}
		else{
			push();
			l->set(ql, qr, x);
			r->set(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	void update(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			if(lset != 	id[2]) lset = lop(lset, x);
			else lazy = lop(lazy, x);
			val = aop(val, x, high - low);
		}
		else{
			push();
			l->update(ql, qr, x);
			r->update(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	T query(int ql, int qr){
		if(qr <= low || high <= ql) return id[1];
		if(ql <= low && high <= qr) return val;
		push();
		return qop(l->query(ql, qr), r->query(ql, qr));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<class IO>
	int plb(T val, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(l->val < val) return r->plb(inv_op(val, l->val), inv_op);
		else return l->plb(val, inv_op);
	}
	template<class IO>
	int lower_bound(T val, IO inv_op){ // min i such that query[0, i) >= val
		if(this->val < val) return high + 1;
		else return plb(val, inv_op);
	}
	template<class IO>
	int lower_bound(int i, T val, IO inv_op){
		return lower_bound(qop(val, query(low, min(i, high))), inv_op);
	}
	template<class IO>
	int pub(T val, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(val < l->val) return l->pub(val, inv_op);
		else return r->pub(inv_op(val, l->val), inv_op);
	}
	template<class IO>
	int upper_bound(T val, IO inv_op){ // min i such that query[0, i) > val
		if(val < this->val) return pub(val, inv_op);
		else return high + 1;
	}
	template<class IO>
	int upper_bound(int i, T val, IO inv_op){
		return upper_bound(qop(val, query(low, min(i, high))), inv_op);
	}
};

// 156485479_3_2_7
// Persistent Segment Tree
// O(N) preprocessing, O(log N) per query
template<class T>
struct node{
	node *l = 0, *r = 0;
	T val;
	node(T val): val(val){}
	node(node *l, node *r, function<T(T, T)> bin_op, T id): l(l), r(r), val(id){
		if(l) val = bin_op(l->val, val);
		if(r) val = bin_op(val, r->val);
	}
};
template<class T, class BO>
struct persistent_segment: vector<node<T> *>{
	int N;
	BO bin_op;
	const T id;
	persistent_segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id){
		this->push_back(build(arr, 0, N));
	}
	node<T> *build(const vector<T> &arr, int left, int right){
		if(left + 1 == right) return new node<T>(arr[left]);
		int mid = left + right >> 1;
		return new node<T>(build(arr, left, mid), build(arr, mid, right), bin_op, id);
	}
	T pq(node<T> *u, int left, int right, int ql, int qr){
		if(qr <= left || right <= ql) return id;
		if(ql <= left && right <= qr) return u->val;
		int mid = left + right >> 1;
		return bin_op(pq(u->l, left, mid, ql, qr), pq(u->r, mid, right, ql, qr));
	}
	T query(node<T> *u, int ql, int qr){
		return pq(u, 0, N, ql, qr);
	}
	node<T> *ps(node<T> *u, int left, int right, int p, int val){
		if(left + 1 == right) return new node<T>(val);
		int mid = left + right >> 1;
		if(mid > p) return new node<T>(ps(u->l, left, mid, p, val), u->r, bin_op, id);
		else return new node<T>(u->l, ps(u->r, mid, right, p, val), bin_op, id);
	}
	void set(node<T> *u, int p, int val){
		this->push_back(ps(u, 0, N, p, val));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<class IO>
	int plb(node<T> *u, int left, int right, T val, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(u->l->val < val) return plb(u->r, mid, right, inv_op(val, u->l->val), inv_op);
		else return plb(u->l, left, mid, val, inv_op);
	}
	template<class IO>
	int lower_bound(node<T> *u, T val, IO inv_op){ // min i such that query[0, i) >= val
		if(u->val < val) return N + 1;
		else return plb(u, 0, N, val, inv_op);
	}
	template<class IO>
	int lower_bound(node<T> *u, int i, T val, IO inv_op){
		return lower_bound(u, bin_op(val, query(u, 0, min(i, N))), inv_op);
	}
	template<class IO>
	int pub(node<T> *u, int left, int right, T val, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(val < u->l->val) return pub(u->l, left, mid, val, inv_op);
		else return pub(u->r, mid, right, inv_op(val, u->l->val), inv_op);
	}
	template<class IO>
	int upper_bound(node<T> *u, T val, IO inv_op){ // min i such that query[0, i) > val
		if(val < u->val) return pub(u, 0, N, val, inv_op);
		else return N + 1;
	}
	template<class IO>
	int upper_bound(node<T> *u, int i, T val, IO inv_op){
		return upper_bound(u, bin_op(val, query(u, 0, min(i, N))), inv_op);
	}
};

// 156485479_3_3_1
// Fenwick Tree
// Only works on a commutative group
// O(N log N) preprocessing, O(log N) per query
template<class T, class BO, class IO>
struct fenwick: vector<T>{
	int N;
	BO bin_op;
	IO inv_op;
	T id;
	fenwick(const vector<T> &arr, BO bin_op, IO inv_op, T id): N(arr.size()), bin_op(bin_op), inv_op(inv_op), id(id){
		this->resize(N + 1, id);
		for(int i = 0; i < N; ++ i) update(i, arr[i]);
	}
	void update(int p, T val){
		for(p ++; p <= N; p += p & -p) (*this)[p] = bin_op((*this)[p], val);
	}
	T sum(int p){
		T res = id;
		for(p ++; p > 0; p -= p & -p) res = bin_op(res, (*this)[p]);
		return res;
	}
	T query(int l, int r){
		return inv_op(sum(r - 1), sum(l - 1));
	}
};

// 156485479_3_3_2
// Fenwick Tree Supporting Range Queries of The Same Type
// O(N log N) preprocessing, O(log N) per query
template<class T, class BO, class IO, class MO>
struct rangefenwick{
	fenwick<T, BO, IO> tr1, tr2;
	BO bin_op;
	IO inv_op;
	MO multi_op;
	T id;
	rangefenwick(int N, BO bin_op, IO inv_op, MO multi_op, T id):
		tr1(vector<T>(N, id), bin_op, inv_op, id),
		tr2(vector<T>(N, id), bin_op, inv_op, id),
		bin_op(bin_op), inv_op(inv_op), id(id){}
	void update(int l, int r, T val){
		tr1.update(l, val);
		tr1.update(r, inv_op(id, val));
		tr2.update(l, multi_op(val, l - 1));
		tr2.update(r, inv_op(id, multi_op(val, r - 1)));
	}
	T sum(int p){
		return inv_op(multi_op(tr1.sum(p), p), tr2.sum(p));
	}
	T query(int l, int r){
		return inv_op(sum(r - 1), sum(l - 1));
	}
};

// 156485479_3_3_3
// 2D Fenwick Tree ( Only for Commutative Group )
// O(NM log NM) preprocessing, O(log N log M) per query
template<class T, class BO, class IO>
struct fenwick2d: vector<vector<T>>{
	int N, M;
	BO bin_op;
	IO inv_op;
	T id;
	fenwick2d(const vector<vector<T>> &arr, BO bin_op, IO inv_op, T id): N(sz(arr)), M(sz(arr[0])), bin_op(bin_op), inv_op(inv_op), id(id){
		this->resize(N + 1, vector<T>(M + 1));
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) update(i, j, arr[i][j]);
	}
	void update(int x, int y, T val){
		x ++, y ++;
		for(int i = x; i <= N; i += i & -i) for(int j = y; j <= N; j += j & -j) (*this)[i][j] = bin_op((*this)[i][j], val);
	}
	T sum(int x, int y){
		T res = id;
		++ x, ++ y;
		for(int i = x; i > 0; i -= i & -i) for(int j = y; j > 0; j -= j & -j) res = bin_op(res, (*this)[i][j]);
		return res;
	}
	T query(int x1, int y1, int x2, int y2){
		-- x1, -- y1, -- x2, -- y2;
		return inv_op(bin_op(sum(x2, y2), sum(x1, y1)), bin_op(sum(x2, y1), sum(x1, y2)));
	}
};

// 156485479_3_4
// Wavelet Tree ( WARNING: NOT THOROUGHLY TESTED YET )
// O(L log N) preprocessing, O(log N) per query
struct node: vi{
	int N, low, high;
	node *l = NULL, *r = NULL;
	node(vi::iterator bg, vi::iterator ed, int low, int high, function<bool(int, int)> cmp):
		N(ed - bg), low(low), high(high){
		if(!N) return;
		if(low + 1 == high){
			this->resize(N + 1);
			iota(this->begin(), this->end(), 0);
			return;
		}
		int mid = low + high >> 1;
		auto pred = [&](int x){return cmp(x, mid);};
		this->reserve(N + 1);
		this->push_back(0);
		for(auto it = bg; it != ed; it ++){
			this->push_back(this->back() + pred(*it));
		}
		auto p = stable_partition(bg, ed, pred);
		l = new node(bg, p, low, mid, cmp);
		r = new node(p, ed, mid, high, cmp);
	}
};
struct wavelet{
	int N;
	node *root;
	function<bool(int, int)> cmp;
	vi arr;
	wavelet(const vi &other, function<bool(int, int)> cmp = less<int>()):
		N(other.size()), arr(other), cmp(cmp){
		root = new node(all(arr), *min_element(all(arr), cmp), *max_element(all(arr), cmp) + 1, cmp);
	}
	//Count elements less than val in the range [l, r)
	int count(node *node, int ql, int qr, int val){
		if(ql >= qr || !cmp(node->low, val)) return 0;
		if(!cmp(val, node->high)) return qr - ql;
		int Lcnt = (*node)[ql], Rcnt = (*node)[qr];
		return count(node->l, Lcnt, Rcnt, val) + count(node->r, ql - Lcnt, qr - Rcnt, val);
	}
	//Find the kth element in the range [l, r)
	int kth(node *node, int ql, int qr, int k){
		if(k > node->N) return node->high;
		if(k <= 0) return node->low - 1;
		if(node->low + 1 == node->high) return node->low;
		int Lcnt = (*node)[ql], Rcnt = (*node)[qr];
		if(k <= node->l->N) return kth(node->l, Lcnt, Rcnt, k);
		else return kth(node->r, ql - Lcnt, qr - Rcnt, k - node->l->N);
	}
};

// 156485479_3_5
// Disjoint Set
// O(alpha(n)) per query where alpha(n) is the inverse ackermann function
struct disjoint: vi{
	int N;
	// vector<pii> Log; // For persistency
	disjoint(int N): N(N){
		this->resize(N);
		iota(all((*this)), 0);
	}
	int root(int u){
		// Log.emplace_back(u, (*this)[u]);
		return (*this)[u] == u ? u : ((*this)[u] = root((*this)[u]));
	}
	void merge(int u, int v){
		(*this)[root(v)] = root(u);
	}
	int share(int u, int v){
		return root((*this)[u]) == root((*this)[v]);
	}
	/*void reverse(){
		auto [u, p] = Log.back();
		Log.pop_back();
		(*this)[u] = p;
	}*/
};

// 156485479_3_6
// Monotone Stack
// O(1) per operation
template<class T = int, class Compare = function<bool(T, T)>>
struct monotone_stack: vector<T>{
	T init;
	Compare cmp;
	monotone_stack(T init = 0, Compare cmp = less<T>{}): init(init), cmp(cmp){ }
	T push(T x){
		while(!this->empty() && !cmp(this->back(), x)) this->pop_back();
		this->push_back(x);
		return this->size() == 1 ? init : *-- -- this->end();
	}
};

// 156485479_3_7
// Persistent Array
// O(N) initalization, O(log N) per operations
template<class T>
struct parray{
	int k;
	T val;
	parray<T> *left, *right;
	parray(int k, T val, parray<T>* left, parray<T>* right): k(k), val(val), left(left), right(right){ }
	static parray<T>* create(int i, int j, T init){
		if(i > j) return 0;
		else{
			int k = (i + j) / 2;
			return new parray<T>(k, init, create(i, k-1, init), create(k+1, j, init));
		}
	}
	T get(int i){
		if (i == k) return val;
		else if (i < k) return left->get(i);
		else return right->get(i);
	}
	parray<T>* set(int i, T new_val){
		if (i == k) return new parray<T>(k, new_val, left, right);
		else if (i < k) return new parray<T>(k, val, left->set(i, new_val), right);
		else return new parray<T>(k, val, left, right->set(i, new_val));
	}
};

// 156485479_3_8
// Persistent Disjoint Set
// O(N) initalization, O(log^2 N) per operations
template<class T>
struct parray{
	int k;
	T val;
	parray<T> *left, *right;
	parray(int k, T val, parray<T>* left, parray<T>* right): k(k), val(val), left(left), right(right){}
	static parray<T>* create(int i, int j, T init){
		if(i > j) return 0;
		else{
			int k = (i + j) / 2;
			return new parray<T>(k, init, create(i, k-1, init), create(k+1, j, init));
		}
	}
	T get(int i){
		if (i == k) return val;
		else if (i < k) return left->get(i);
		else return right->get(i);
	}
	parray<T>* set(int i, T new_val){
		if (i == k) return new parray<T>(k, new_val, left, right);
		else if (i < k) return new parray<T>(k, val, left->set(i, new_val), right);
		else return new parray<T>(k, val, left, right->set(i, new_val));
	}
};
struct pdisjoint{
	parray<int> *parent, *rank;
	pdisjoint(parray<int>* parent, parray<int>* rank): parent(parent), rank(rank){}
	static pdisjoint* create(int n){
		return new pdisjoint(parray<int>::create(0, n-1, -1), parray<int>::create(0, n-1, 0));
	}
	int root(int i){
		int j = parent->get(i);
		return j == -1 ? i : root(j);
	}
	pdisjoint* merge(int i, int j){
		int fi = root(i), fj = root(j);
		if(fi == fj) return this;
		else{
			int ri = rank->get(fi), rj = rank->get(fj);
			if(ri < rj) return new pdisjoint(parent->set(fi, fj), rank);
			else if (ri > rj) return new pdisjoint(parent->set(fj, fi), rank);
			else return new pdisjoint(parent->set(fi, fj), rank->set(fj, rj + 1));
		}
	}
};

// 156485479_3_9
// Less-than-k Query, Distinct Value Query (Offline, Online)
// O(N log N) processing
template<class T, class BO, class IO>
struct fenwick: vector<T>{
	int N;
	BO bin_op;
	IO inv_op;
	T id;
	fenwick(const vector<T> &arr, BO bin_op, IO inv_op, T id): N(arr.size()), bin_op(bin_op), inv_op(inv_op), id(id){
		this->resize(N + 1, id);
		for(int i = 0; i < N; ++ i) update(i, arr[i]);
	}
	void update(int p, T val){
		for(p ++; p <= N; p += p & -p) (*this)[p] = bin_op((*this)[p], val);
	}
	T sum(int p){
		T res = id;
		for(p ++; p > 0; p -= p & -p) res = bin_op(res, (*this)[p]);
		return res;
	}
	T query(int l, int r){
		return inv_op(sum(r - 1), sum(l - 1));
	}
};
template<class T>
struct offline_less_than_k_query{
	int N;
	vector<pair<T, int>> event;
	vector<tuple<T, int, int, int>> queries;
	offline_less_than_k_query(const vector<T> &arr, bool IS_DVQ = true): N(sz(arr)), event(N){
		if(IS_DVQ){
			map<T, int> q;
			for(int i = 0; i < N; ++ i){
				event[i] = {(q.count(arr[i]) ? q[arr[i]] : -1), i};
				q[arr[i]] = i;
			}
		}
		else for(int i = 0; i < N; ++ i) event[i] = {arr[i], i};
	}
	void query(int i, int ql, int qr){ // For distinct value query
		queries.emplace_back(ql, ql, qr, i);
	}
	void query(int i, int ql, int qr, T k){ // For less-than-k query
		queries.emplace_back(k, ql, qr, i);
	}
	template<class Action>
	void solve(Action ans){ // ans(index, answer)
		sort(all(queries)), sort(all(event), greater<pair<T, int>>());
		fenwick tr(vi(N), plus<int>(), minus<int>(), 0);
		for(auto &[k, ql, qr, i]: queries){
			while(!event.empty() && event.back().first < k){
				tr.update(event.back().second, 1);
				event.pop_back();
			}
			ans(i, tr.query(ql, qr));
		}
	}
};


// Online
template<class T>
struct node{
	node *l = 0, *r = 0;
	T val;
	node(T val): val(val){}
	node(node *l, node *r, function<T(T, T)> bin_op, T id): l(l), r(r), val(id){
		if(l) val = bin_op(l->val, val);
		if(r) val = bin_op(val, r->val);
	}
};
template<class T, class BO>
struct persistent_segment: vector<node<T> *>{
	int N;
	BO bin_op;
	const T id;
	persistent_segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id){
		this->push_back(build(arr, 0, N));
	}
	node<T> *build(const vector<T> &arr, int left, int right){
		if(left + 1 == right) return new node<T>(arr[left]);
		int mid = left + right >> 1;
		return new node<T>(build(arr, left, mid), build(arr, mid, right), bin_op, id);
	}
	T pq(node<T> *u, int left, int right, int ql, int qr){
		if(qr <= left || right <= ql) return id;
		if(ql <= left && right <= qr) return u->val;
		int mid = left + right >> 1;
		return bin_op(pq(u->l, left, mid, ql, qr), pq(u->r, mid, right, ql, qr));
	}
	T query(node<T> *u, int ql, int qr){
		return pq(u, 0, N, ql, qr);
	}
	node<T> *ps(node<T> *u, int left, int right, int p, int val){
		if(left + 1 == right) return new node<T>(val);
		int mid = left + right >> 1;
		if(mid > p) return new node<T>(ps(u->l, left, mid, p, val), u->r, bin_op, id);
		else return new node<T>(u->l, ps(u->r, mid, right, p, val), bin_op, id);
	}
	void set(node<T> *u, int p, int val){
		this->push_back(ps(u, 0, N, p, val));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<class IO>
	int plb(node<T> *u, int left, int right, T val, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(u->l->val < val) return plb(u->r, mid, right, inv_op(val, u->l->val), inv_op);
		else return plb(u->l, left, mid, val, inv_op);
	}
	template<class IO>
	int lower_bound(node<T> *u, T val, IO inv_op){ // min i such that query[0, i) >= val
		if(u->val < val) return N + 1;
		else return plb(u, 0, N, val, inv_op);
	}
	template<class IO>
	int lower_bound(node<T> *u, int i, T val, IO inv_op){
		return lower_bound(u, bin_op(val, query(u, 0, min(i, N))), inv_op);
	}
	template<class IO>
	int pub(node<T> *u, int left, int right, T val, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(val < u->l->val) return pub(u->l, left, mid, val, inv_op);
		else return pub(u->r, mid, right, inv_op(val, u->l->val), inv_op);
	}
	template<class IO>
	int upper_bound(node<T> *u, T val, IO inv_op){ // min i such that query[0, i) > val
		if(val < u->val) return pub(u, 0, N, val, inv_op);
		else return N + 1;
	}
	template<class IO>
	int upper_bound(node<T> *u, int i, T val, IO inv_op){
		return upper_bound(u, bin_op(val, query(u, 0, min(i, N))), inv_op);
	}
};
template<class T>
struct less_than_k_query{ // for less-than-k query, it only deals with numbers in range [0, N)
	int N;
	vector<node<T> *> p;
	persistent_segment<int, plus<int>> tr;
	less_than_k_query(const vector<T> &arr, bool IS_DVQ = true): N(sz(arr)), p(N + 1), tr(vi(N), plus<int>{}, 0){
		vector<pair<T, int>> event(N);
		if(IS_DVQ){
			map<T, int> q;
			for(int i = 0; i < N; ++ i){
				event[i] = {(q.count(arr[i]) ? q[arr[i]] : -1), i};
				q[arr[i]] = i;
			}
		}
		else for(int i = 0; i < N; ++ i) event[i] = {arr[i], i};
		sort(all(event), greater<pii>{});
		tr.reserve(N);
		for(int i = 0; i <= N; ++ i){
			while(!event.empty() && event.back().first < i){
				tr.set(tr.back(), event.back().second, 1);
				event.pop_back();
			}
			p[i] = tr.back();
		}
	}
	// For distinct value query
	int query(int ql, int qr){
		return tr.query(p[ql], ql, qr);
	}
	int lower_bound(int ql, int cnt){ // min i such that # of distinct in [l, l + i) >= cnt
		return tr.lower_bound(p[ql], ql, cnt, minus<int>());
	}
	int upper_bound(int ql, int cnt){ // min i such that # of distinct in [l, l + i) > cnt
		return tr.upper_bound(p[ql], ql, cnt, minus<int>());
	}
	// For less-than-k query
	int query(int ql, int qr, int k){
		return tr.query(p[k], ql, qr);
	}
	int lower_bound(int ql, int k, int cnt){ // min i such that ( # of elements < k in [l, l + i) ) >= cnt
		return tr.lower_bound(p[k], ql, cnt, minus<int>());
	}
	int upper_bound(int ql, int k, int cnt){ // min i such that ( # of elements < k in [l, l + i) ) > cnt
		return tr.upper_bound(p[k], ql, cnt, minus<int>());
	}
};

// 156485479_3_10
// Mo's Algorithm
// O((N + Q) sqrt(N) F) where F is the processing time of ins and del.
template<int B>
struct Query{
	int l, r, ind;
	bool operator<(const Query &otr) const{
		if(l / B != otr.l / B) return pii(l, r) < pii(otr.l, otr.r);
		return (l / B & 1) ? (r < otr.r) : (r > otr.r);
	}
};
template<class T, class Q, class I, class D, class A>
vector<T> answer_query_offline(const vector<T> &arr, vector<Q> query, I ins, D del, A ans){
	sort(all(query));
	vector<T> res(sz(query));
	for(auto q: query){
		static int l = 0, r = 0;
		while(l > q.l) ins(-- l);
		while(r < q.r) ins(r ++);
		while(l < q.l) del(l ++);
		while(r > q.r) del(-- r);
		res[q.ind] = ans();
	}
	return res;
}

// 156485479_4_1
// Strongly Connected Component ( Tarjan's Algorithm )
// O(n + m)
template<class G, class F>
int scc(const G &g, F f){
	int n = sz(g);
	vi val(n, 0), comp(n, -1), z, cur;
	int Time = 0, ncomps = 0;
	auto dfs = [&](int u, auto &dfs)->int{
		int low = val[u] = ++ Time, v;
		z.push_back(u);
		for(auto v: g[u]) if(comp[v] < 0) low = min(low, val[v] ?: dfs(v, dfs));
		if(low == val[u]){
			do{
				v = z.back(); z.pop_back();
				comp[v] = ncomps;
				cur.push_back(v);
			}while(v != u);
			f(cur); // Process SCCs in reverse topological order
			cur.clear();
			ncomps ++;
		}
		return val[u] = low;
	};
	for(int u = 0; u < n; ++ u) if(comp[u] < 0) dfs(u, dfs);
	return ncomps;
}

// 156485479_4_2
// Biconnected Components
// O(n + m)
template<class G, class F, class FF>
int bcc(const G &g, F f, FF ff = [](int u, int v, int e){}){
	int n = sz(g);
	vi num(n), st;
	int Time = 0, ncomps = 0;
	auto dfs = [&](int u, int pe, auto &dfs)->int{
		int me = num[u] = ++ Time, top = me;
		for(auto [v, e]: g[u]) if(e != pe){
			if(num[v]){
				top = min(top, num[v]);
				if(num[v] < me) st.push_back(e);
			}
			else{
				int si = sz(st);
				int up = dfs(v, e, dfs);
				top = min(top, up);
				if(up == me){
					st.push_back(e);
					f(vi(st.begin() + si, st.end())); // Process BCCs (edgelist)
					st.resize(si);
					ncomps ++;
				}
				else if(up < me) st.push_back(e);
				else ff(u, v, e); // Process bridges
			}
		}
		return top;
	};
	for(int u = 0; u < n; ++ u) if(!num[u]) dfs(u, -1, dfs);
	return ncomps;
}

// 156485479_4_3_1
// Dinic's Maximum Flow Algorithm
// O(V^2E) ( O(E*sqrt(V)) for unit network )
template<class T>
struct flownetwork{
	static constexpr T eps = (T)1e-9;
	int n;
	vector<vector<int>> adj;
	struct edge{
		int from, to;
		T capacity, flow;
	};
	vector<edge> edge;
	int source, sink;
	T flow;
	flownetwork(int n, int source, int sink):
		n(n), source(source), sink(sink){
		adj.resize(n);
		flow = 0;
	}
	void clear(){
		for(auto &e: edge) e.flow = 0;
		flow = 0;
	}
	int insert(int from, int to, T fcap, T bcap){
		int ind = edge.size();
		adj[from].push_back(ind);
		edge.push_back({from, to, fcap, 0});
		adj[to].push_back(ind + 1);
		edge.push_back({to, from, bcap, 0});
		return ind;
	}
};
template<class T>
struct dinic{
	static constexpr T INF = numeric_limits<T>::max();
	flownetwork<T> &g;
	vi ptr, level, q;
	dinic(flownetwork<T> &g): g(g){
		ptr.resize(g.n), level.resize(g.n), q.resize(g.n);
	}
	bool bfs(){
		fill(all(level), -1);
		q[0] = g.sink;
		level[g.sink] = 0;
		int beg = 0, end = 1;
		while(beg < end){
			int i = q[beg ++];
			for(auto ind: g.adj[i]){
				auto &e = g.edge[ind];
				auto &re = g.edge[ind ^ 1];
				if(re.capacity - re.flow > g.eps && level[e.to] == -1){
					level[e.to] = level[i] + 1;
					if(e.to == g.source) return true;
					q[end ++] = e.to;
				}
			}
		}
		return false;
	}
	T dfs(int u, T w){
		if(u == g.sink) return w;
		int &j = ptr[u];
		while(j >= 0){
			int ind = g.adj[u][j];
			auto &e = g.edge[ind];
			if(e.capacity - e.flow > g.eps && level[e.to] == level[u] - 1){
				T F = dfs(e.to, min(e.capacity - e.flow, w));
				if(F > g.eps){
					g.edge[ind].flow += F;
					g.edge[ind ^ 1].flow -= F;
					return F;
				}
			}
			j --;
		}
		return 0;
	}
	T max_flow(){
		while(bfs()){
			for(int i = 0; i < g.n; ++ i) ptr[i] = g.adj[i].size() - 1;
			T sum = 0;
			while(1){
				T add = dfs(g.source, INF);
				if(add <= g.eps) break;
				sum += add;
			}
			if(sum <= g.eps) break;
			g.flow += sum;
		}
		return g.flow;
	}
	vector<bool> min_cut(){
		max_flow();
		vector<bool> res(g.n);
		for(int i = 0; i < g.n; i ++) res[i] = (level[i] != -1);
		return res;
	}
};

// 156485479_4_3_2
// Minimum Cost Maximum Flow Algorithm
// O(N^2 M^2)
template<typename T, typename C>
struct mcmf{
	static constexpr T eps = (T) 1e-9;
	struct edge{
		int from, to;
		T c, f;
		C cost;
	};
	vector<vi> g;
	vector<edge> edges;
	vector<C> d;
	vector<bool> in_queue;
	vi q, pe;
	int n, source, sink;
	T flow;
	C cost;
	mcmf(int n, int source, int sink): n(n), source(source), sink(sink), g(n), d(n), in_queue(n), pe(n){
		assert(0 <= source && source < n && 0 <= sink && sink < n && source != sink);
		flow = cost = 0;
	}
	void clear_flow(){
		for(const edge &e: edges) e.f = 0;
		flow = 0;
	}
	void add(int from, int to, T forward_cap, T backward_cap, C cost){
		assert(0 <= from && from < n && 0 <= to && to < n);
		g[from].push_back((int) edges.size());
		edges.push_back({from, to, forward_cap, 0, cost});
		g[to].push_back((int) edges.size());
		edges.push_back({to, from, backward_cap, 0, -cost});
	}
	bool expath(){
		fill(d.begin(), d.end(), numeric_limits<C>::max());
		q.clear();
		q.push_back(source);
		d[source] = 0;
		in_queue[source] = true;
		int beg = 0;
		bool found = false;
		while(beg < q.size()){
			int i = q[beg ++];
			if(i == sink) found = true;
			in_queue[i] = false;
			for(int id : g[i]){
				const edge &e = edges[id];
				if(e.c - e.f > eps && d[i] + e.cost < d[e.to]){
					d[e.to] = d[i] + e.cost;
					pe[e.to] = id;
					if(!in_queue[e.to]){
						q.push_back(e.to);
						in_queue[e.to] = true;
					}
				}
			}
		}
		if(found){
			T push = numeric_limits<T>::max();
			int v = sink;
			while(v != source){
				const edge &e = edges[pe[v]];
				push = min(push, e.c - e.f);
				v = e.from;
			}
			v = sink;
			while(v != source){
				edge &e = edges[pe[v]];
				e.f += push;
				edge &back = edges[pe[v] ^ 1];
				back.f -= push;
				v = e.from;
			}
			flow += push;
			cost += push * d[sink];
		}
		return found;
	}
	pair<T, C> get_mcmf(){
		while(expath()){ }
		return {flow, cost};
	}
};

// 156485479_4_4_1
// LCA
// O(N log N) precomputing, O(1) per query
template<class T, class BO = function<T(T, T)>>
struct sparse_table: vector<vector<T>>{
	int N;
	BO bin_op;
	sparse_table(const vector<T> &arr, BO bin_op = [](T x, T y){return min(x, y);}): N(arr.size()), bin_op(bin_op){
		int t = 1, d = 1;
		while(t < N) t *= 2, ++ d;
		this->assign(d, arr);
		for(int i = 0; i < d - 1; ++ i) for(int j = 0; j < N; ++ j){
			(*this)[i + 1][j] = bin_op((*this)[i][j], (*this)[i][min(N - 1, j + (1 << i))]);
		}
	}
	sparse_table(){ }
	T query(int l, int r){
		int d = 31 - __builtin_clz(r - l);
		return bin_op((*this)[d][l], (*this)[d][r - (1 << d)]);
	}
	sparse_table &operator=(const sparse_table &otr){
		N = otr.N, bin_op = otr.bin_op; this->clear();
		for(auto &t: otr) this->push_back(t);
		return *this;
	}
};
struct LCA{
	vi time;
	vl depth;
	int root;
	sparse<pii> rmq;
	LCA(vector<vector<pii>> &adj, int root): root(root), time(adj.size(), -99), depth(adj.size()), rmq(dfs(adj)){}
	vector<pii> dfs(vector<vector<pii>> &adj){
		vector<tuple<int, int, int, ll>> q(1);
		vector<pii> res;
		int T = root;
		while(!q.empty()){
			auto [u, p, d, di] = q.back();
			q.pop_back();
			if(d) res.emplace_back(d, p);
			time[u] = T ++;
			depth[u] = di;
			for(auto &e: adj[u]) if(e.first != p){
				q.emplace_back(e.first, u, d + 1, di + e.second);
			}
		}
		return res;
	}
	int query(int l, int r){
		if(l == r) return l;
		l = time[l], r = time[r];
		return rmq.query(min(l, r), max(l, r)).second;
	}
	ll dist(int l, int r){
		int lca = query(l, r);
		return depth[l] + depth[r] - 2 * depth[lca];
	}
};

// 156485479_4_4_2_1
// Binary Lifting for Unweighted Tree
// O(N log N) preprocessing, O(log N) per lca query
struct binary_lift: vector<vi>{
	int N, root, lg;
	vector<vi> up;
	vi depth;
	binary_lift(int N, int root): N(N), root(root), lg(ceil(log2(N))), depth(N), up(N, vector<int>(lg + 1)){
		this->resize(N);
	}
	void insert(int u, int v){
		(*this)[u].push_back(v);
		(*this)[v].push_back(u);
	}
	void init(){
		dfs(root, root);
	}
	void dfs(int u, int p){
		up[u][0] = p;
		for(int i = 1; i <= lg; ++ i) up[u][i] = up[up[u][i - 1]][i - 1];
		for(auto &v: (*this)[u]) if(v != p){
			depth[v] = depth[u] + 1;
			dfs(v, u);
		}
	}
	int lca(int u, int v){
		if(depth[u] < depth[v]) std::swap(u, v);
		u = trace_up(u, depth[u] - depth[v]);
		for(int d = lg; d >= 0; -- d) if(up[u][d] != up[v][d]) u = up[u][d], v = up[v][d];
		return u == v ? u : up[u][0];
	}
	int dist(int u, int v){
		return depth[u] + depth[v] - 2 * depth[lca(u, v)];
	}
	int trace_up(int u, int dist){
		if(dist >= depth[u] - depth[root]) return root;
		for(int d = lg; d >= 0; -- d) if(dist & (1 << d)) u = up[u][d];
		return u;
	}
};

// 156485479_4_4_2_2
// Binary Lifting for Weighted Tree Supporting Commutative Monoid Operations
// O(N log N) processing, O(log N) per query
template<class T, class BO>
struct binary_lift: vector<vector<pair<int, T>>>{
	int N, root, lg;
	BO bin_op;
	T id;
	vector<vector<pair<int, T>>> up;
	const vector<T> &val;
	vi depth;
	binary_lift(int N, int root, const vector<T> &val, BO bin_op, T id): N(N), root(root), bin_op(bin_op), id(id), lg(32 - __builtin_clz(N)), depth(N), val(val), up(N, vector<pair<int, T>>(lg + 1)){
		this->resize(N);
	}
	void insert(int u, int v, T w){ // w = id if no edge weight
		(*this)[u].push_back({v, w});
		(*this)[v].push_back({u, w});
	}
	void init(){
		dfs(root, root, id);
	}
	void dfs(int u, int p, T w){
		up[u][0] = {p, bin_op(val[u], w)};
		for(int i = 1; i <= lg; ++ i) up[u][i] = {
			up[up[u][i - 1].first][i - 1].first
			, bin_op(up[u][i - 1].second, up[up[u][i - 1].first][i - 1].second)
		};
		for(auto &[v, x]: (*this)[u]) if(v != p){
			depth[v] = depth[u] + 1;
			dfs(v, u, x);
		}
	}
	pair<int, T> trace_up(int u, int dist){ // Node, Distance (Does not include weight of the node)
		T res = id;
		dist = min(dist, depth[u] - depth[root]);
		for(int d = lg; d >= 0; -- d) if(dist & (1 << d)){
			res = bin_op(res, up[u][d].second), u = up[u][d].first;
		}
		return {u, res};
	}
	pair<int, T> query(int u, int v){ // LCA, Query Value
		if(depth[u] < depth[v]) swap(u, v);
		T res;
		tie(u, res) = trace_up(u, depth[u] - depth[v]);
		for(int d = lg; d >= 0; -- d) if(up[u][d].first != up[v][d].first){
			res = bin_op(res, up[u][d].second), u = up[u][d].first;
			res = bin_op(res, up[v][d].second), v = up[v][d].first;
		}
		if(u != v) res = bin_op(bin_op(res, up[u][0].second), up[v][0].second), u = up[u][0].first;
		return {u, bin_op(res, val[u])};
	}
	int dist(int u, int v){
		return depth[u] + depth[v] - 2 * depth[query(u, v).first];
	}
};

// 156485479_4_4_3
// Heavy Light Decomposition
// O(N + M) processing, O(log^2 N) per query
template<class T, class BO1, class BO2, class BO3>
struct lazy_segment{
	lazy_segment *l = 0, *r = 0;
	int low, high;
	BO1 lop;           // Lazy op(L, L -> L)
	BO2 qop;           // Query op(Q, Q -> Q)
	BO3 aop;           // Apply op(Q, L, len -> Q)
	vector<T> &id;     // Lazy id(L), Query id(Q), Disable constant(Q)
	T lset, lazy, val;
	lazy_segment(int low, int high, BO1 lop, BO2 qop, BO3 aop, vector<T> &id): low(low), high(high), lop(lop), qop(qop), aop(aop), id(id){
		lazy = id[0], val = id[1], lset = id[2];
	}
	lazy_segment(const vector<T> &arr, int low, int high, BO1 lop, BO2 qop, BO3 aop, vector<T> &id)
	: low(low), high(high), lop(lop), qop(qop), aop(aop), id(id){
		lazy = id[0], lset = id[2];
		if(high - low > 1){
			int mid = low + (high - low) / 2;
			l = new lazy_segment(arr, low, mid, lop, qop, aop, id);
			r = new lazy_segment(arr, mid, high, lop, qop, aop, id);
			val = qop(l->val, r->val);
		}
		else val = arr[low];
	}
	void push(){
		if(!l){
			int mid = low + (high - low) / 2;
			l = new lazy_segment(low, mid, lop, qop, aop, id);
			r = new lazy_segment(mid, high, lop, qop, aop, id);
		}
		if(lset != id[2]){
			l->set(low, high, lset);
			r->set(low, high, lset);
			lset = id[2];
		}
		else if(lazy != id[0]){
			l->update(low, high, lazy);
			r->update(low, high, lazy);
			lazy = id[0];
		}
	}
	void set(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			lset = x;
			lazy = id[0];
			val = aop(id[1], x, high - low);
		}
		else{
			push();
			l->set(ql, qr, x);
			r->set(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	void update(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			if(lset != 	id[2]) lset = lop(lset, x);
			else lazy = lop(lazy, x);
			val = aop(val, x, high - low);
		}
		else{
			push();
			l->update(ql, qr, x);
			r->update(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	T query(int ql, int qr){
		if(qr <= low || high <= ql) return id[1];
		if(ql <= low && high <= qr) return val;
		push();
		return qop(l->query(ql, qr), r->query(ql, qr));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<class IO>
	int plb(T val, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(l->val < val) return r->plb(inv_op(val, l->val), inv_op);
		else return l->plb(val, inv_op);
	}
	template<class IO>
	int lower_bound(T val, IO inv_op){ // min i such that query[0, i) >= val
		if(this->val < val) return high + 1;
		else return plb(val, inv_op);
	}
	template<class IO>
	int lower_bound(int i, T val, IO inv_op){
		return lower_bound(qop(val, query(low, min(i, high))), inv_op);
	}
	template<class IO>
	int pub(T val, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(val < l->val) return l->pub(val, inv_op);
		else return r->pub(inv_op(val, l->val), inv_op);
	}
	template<class IO>
	int upper_bound(T val, IO inv_op){ // min i such that query[0, i) > val
		if(val < this->val) return pub(val, inv_op);
		else return high + 1;
	}
	template<class IO>
	int upper_bound(int i, T val, IO inv_op){
		return upper_bound(qop(val, query(low, min(i, high))), inv_op);
	}
};
template<class DS, class BO, class T, int VALS_IN_EDGES = 1>
struct HLD{
	int N, root;
	vector<vi> adj;
	vi par, size, depth, next, pos, rpos;
	DS &tr;
	BO bin_op;
	const T id;
	HLD(int N, int root, DS &tr, BO bin_op, T id):
	N(N), root(root), adj(N), par(N, -1), size(N, 1), depth(N), next(N), pos(N), tr(tr), bin_op(bin_op), id(id){
		this->root = next[root] = root;
	}
	void insert(int u, int v){
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	void dfs_sz(int u){
		if(par[u] != -1) adj[u].erase(find(all(adj[u]), par[u]));
		for(auto &v: adj[u]){
			par[v] = u, depth[v] = depth[u] + 1;
			dfs_sz(v);
			size[u] += size[v];
			if(size[v] > size[adj[u][0]]) swap(v, adj[u][0]);
		}
	}
	void dfs_hld(int u){
		static int t = 0;
		pos[u] = t ++;
		rpos.push_back(u);
		for(auto &v: adj[u]){
			next[v] = (v == adj[u][0] ? next[u] : v);
			dfs_hld(v);
		}
	}
	void init(){
		dfs_sz(root), dfs_hld(root);
	}
	template<class Action>
	void processpath(int u, int v, Action act){
		for(; next[u] != next[v]; v = par[next[v]]){
			if(depth[next[u]] > depth[next[v]]) swap(u, v);
			act(pos[next[v]], pos[v] + 1);
		}
		if(depth[u] > depth[v]) swap(u, v);
		act(pos[u] + VALS_IN_EDGES, pos[v] + 1);
	}
	void updatepath(int u, int v, T val, int is_update = true){
		if(is_update) processpath(u, v, [this, &val](int l, int r){tr.update(l, r, val);});
		else processpath(u, v, [this, &val](int l, int r){tr.set(l, r, val);});
	}
	void updatesubtree(int u, T val, int is_update = true){
		if(is_update) tr.update(pos[u] + VALS_IN_EDGES, pos[u] + size[u], val);
		else tr.set(pos[u] + VALS_IN_EDGES, pos[u] + size[u], val);
	}
	T querypath(int u, int v){
		T res = id;
		processpath(u, v, [this, &res](int l, int r){res = bin_op(res, tr.query(l, r));});
		return res;
	}
	T querysubtree(int u){
		return tr.query(pos[u] + VALS_IN_EDGES, pos[u] + size[u]);
	}
};

// 156485479_4_4_4
// Centroid Decomposition
// O(N log N) processing
struct CD: vector<vi>{
	int N, root;
	vi dead, size, par, cpar;
	vector<vi> cchild, dist;
	CD(int N): N(N), dead(N), size(N), par(N), cchild(N), cpar(N), dist(N){
		this->resize(N);
	}
	void insert(int u, int v){
		(*this)[u].push_back(v);
		(*this)[v].push_back(u);
	}
	void dfs_sz(int u){
		size[u] = 1;
		for(auto v: (*this)[u]) if(!dead[v] && v != par[u]){
			par[v] = u;
			dfs_sz(v);
			size[u] += size[v];
		}
	}
	int centroid(int u){
		par[u] = -1;
		dfs_sz(u);
		int size = size[u];
		while(1){
			int w = 0, msz = 0;
			for(auto v: (*this)[u]) if(!dead[v] && v != par[u] && msz < size[v]){
				w = v, msz = size[v];
			}
			if(msz * 2 <= size) return u;
			u = w;
		}
	}
	void dfs_dist(int u, int p){
		dist[u].push_back(dist[p].back() + 1);
		for(auto v: (*this)[u]) if(!dead[v] && v != p) dfs_dist(v, u);
	}
	void dfs_centroid(int u, int p){
		dead[u = centroid(u)] = true;
		cpar[u] = p;
		if(p != -1) cchild[p].push_back(u);
		else root = u;
		dist[u].push_back(0);
		int d = 0;
		for(auto v: (*this)[u]) if(!dead[v]) dfs_dist(v, u);
		for(auto v: (*this)[u]) if(!dead[v]) dfs_centroid(v, u);
	}
	void init(){
		dfs_centroid(0, -1);
	}
};

// 156485479_4_4_5
// AHU Algorithm ( Rooted Tree Isomorphism ) / Tree Isomorphism
// O(n)
void radix_sort(vector<pair<int, vi>> &arr){
	int n = sz(arr), mxval = 0, mxsz = 1 + accumulate(all(arr), 0, [](int x, const pair<int, vi> &y){return max(x, sz(y.second));});
	vector<vi> occur(mxsz);
	for(int i = 0; i < n; ++ i){
		occur[sz(arr[i].second)].push_back(i);
		for(auto x: arr[i].second) mxval = max(mxval, x);
	}
	mxval ++;
	for(int size = 1; size < mxsz; ++ size) for(int d = size - 1; d >= 0; -- d){
		vector<vi> bucket(mxval);
		for(auto i: occur[size]) bucket[arr[i].second[d]].push_back(i);
		occur[size].clear();
		for(auto &b: bucket) for(auto i: b) occur[size].push_back(i);
	}
	vector<pair<int, vi>> res;
	res.reserve(n);
	for(auto &b: occur) for(auto i: b) res.push_back(arr[i]);
	swap(res, arr);
}
bool isomorphic(const vector<vector<vi>> &adj, const vi &root){
	int n = sz(adj[0]);
	if(sz(adj[1]) != n) return false;
	vector<vector<vi>> occur(2);
	vector<vi> depth(2, vi(n)), par(2, vi(n, -1));
	for(int k = 0; k < 2; ++ k){
		function<void(int, int)> dfs = [&](int u, int p){
			par[k][u] = p;
			for(auto v: adj[k][u]) if(v != p){
				depth[k][v] = depth[k][u] + 1;
				dfs(v, u);
			}
		};
		dfs(root[k], -1);
		int mxdepth = 1 + accumulate(all(depth[k]), 0, [](int x, int y){return max(x, y);});
		occur[k].resize(mxdepth);
		for(int u = 0; u < n; ++ u) occur[k][depth[k][u]].push_back(u);
	}
	int mxdepth = sz(occur[0]);
	if(mxdepth != sz(occur[1])) return false;
	for(int d = 0; d < mxdepth; ++ d) if(sz(occur[0][d]) != sz(occur[1][d])) return false;
	vector<vi> label(2, vi(n)), pos(2, vi(n));
	vector<vector<vi>> sorted_list(mxdepth, vector<vi>(2));
	for(int k = 0; k < 2; ++ k){
		sorted_list[mxdepth - 1][k].reserve(sz(occur[k][mxdepth - 1]));
		for(auto u: occur[k][mxdepth - 1]) sorted_list[mxdepth - 1][k].push_back(u);
	}
	for(int d = mxdepth - 2; d >= 0; -- d){
		vector<vector<pair<int, vi>>> tuples(2);
		for(int k = 0; k < 2; ++ k){
			tuples[k].reserve(sz(occur[k][d]));
			for(auto u: occur[k][d]){
				pos[k][u] = sz(tuples[k]);
				tuples[k].emplace_back(u, vi());
			}
			for(auto v: sorted_list[d + 1][k]){
				int u = par[k][v];
				tuples[k][pos[k][u]].second.push_back(label[k][v]);
			}
			radix_sort(tuples[k]);
		}
		for(int i = 0; i < sz(tuples[0]); ++ i) if(tuples[0][i].second != tuples[1][i].second) return false;
		for(int k = 0; k < 2; ++ k){
			int cnt = 0;
			sorted_list[d][k].reserve(sz(occur[k][d]));
			sorted_list[d][k].push_back(tuples[k][0].first);
			for(int i = 1; i < sz(tuples[k]); ++ i){
				int u = tuples[k][i].first;
				label[k][u] = (tuples[k][i - 1].second == tuples[k][i].second ? cnt : ++ cnt);
				sorted_list[d][k].push_back(u);
			}
		}
	}
	return true;
}
vi centroid(const vector<vi> &adj){
	int n = sz(adj);
	vi size(n, 1);
	function<void(int, int)> dfs_sz = [&](int u, int p){
		for(auto v: adj[u]) if(v != p){
			dfs_sz(v, u);
			size[u] += size[v];
		}
	};
	dfs_sz(0, -1);
	function<vi(int, int)> dfs_cent = [&](int u, int p){
		for(auto v: adj[u]) if(v != p && size[v] > n / 2) return dfs_cent(v, u);
		for(auto v: adj[u]) if(v != p && n - size[v] <= n / 2) return vi{u, v};
		return vi{u};
	};
	return dfs_cent(0, -1);
}
bool isomorphic(const vector<vector<vi>> &adj){
	vector<vi> cent{centroid(adj[0]), centroid(adj[1])};
	if(sz(cent[0]) != sz(cent[1])) return false;
	for(auto u: cent[0]) for(auto v: cent[1]) if(isomorphic(adj, vi{u, v})) return true;
	return false;
}

// 156485479_5_1
// Returns the starting position of the lexicographically minimal rotation
// O(n)
int min_rotation(string s){
	int n = sz(s);
	s += s;
	int a = 0;
	for(int b = 0; b < n; ++ b) for(int i = 0; i < n; ++ i){
		if(a + i == b || s[a + i] < s[b + i]){
			b += max(0, i - 1);
			break;
		}
		if(s[a + i] > s[b + i]){
			a = b;
			break;
		}
	}
	return a;
}

// 156485479_5_2
// All Palindromic Substrings ( Manachar's Algorithm )
// O(N)
struct manachar{
	int N;
	vector<int> o, e;
	pair<int, int> build(const string &s){
		N = sz(s), o.resize(N), e.resize(N);
		int res = 0, resl, resr;
		int l = 0, r = -1;
		for(int i = 0; i < N; ++ i){
			int k = (i > r) ? 1 : min(o[l + r - i], r - i) + 1;
			while(i - k >= 0 && i + k < N && s[i - k] == s[i + k]) k ++;
			o[i] = -- k;
			if(res < 2 * k + 1){
				res = 2 * k + 1;
				resl = i - k, resr = i + k + 1;
			}
			if(r < i + k){
				l = i - k;
				r = i + k;
			}
		}
		l = 0; r = -1;
		for(int i = 0; i < N; ++ i){
			int k = (i > r) ? 1 : min(e[l + r - i + 1], r - i + 1) + 1;
			while(i - k >= 0 && i + k - 1 < N && s[i - k] == s[i + k - 1]) k ++;
			e[i] = -- k;
			if(res < 2 * k){
				res = 2 * k;
				resl = i - k, resr = i + k;
			}
			if(r < i + k - 1){
				l = i - k;
				r = i + k - 1;
			}
		}
		return {resl, resr};
	}
}mnc;

// 156485479_5_3
// Suffix Array and Kasai's Algorithm
// O(N log N)
template<class T, class BO = function<T(T, T)>>
struct sparse_table: vector<vector<T>>{
	int N;
	BO bin_op;
	sparse_table(const vector<T> &arr, BO bin_op = [](T x, T y){return min(x, y);}): N(arr.size()), bin_op(bin_op){
		int t = 1, d = 1;
		while(t < N) t *= 2, ++ d;
		this->assign(d, arr);
		for(int i = 0; i < d - 1; ++ i) for(int j = 0; j < N; ++ j){
			(*this)[i + 1][j] = bin_op((*this)[i][j], (*this)[i][min(N - 1, j + (1 << i))]);
		}
	}
	sparse_table(){}
	T query(int l, int r){
		int d = 31 - __builtin_clz(r - l);
		return bin_op((*this)[d][l], (*this)[d][r - (1 << d)]);
	}
	sparse_table &operator=(const sparse_table &otr){
		N = otr.N, bin_op = otr.bin_op; this->clear();
		for(auto &t: otr) this->push_back(t);
		return *this;
	}
};
template<class Str, int lim = 256>
struct suffix_array{
	int N;
	Str s;
	vi p, c, l; // p[i]: starting index of i-th suffix in SA, c[i]: position of suffix of index i in SA
	sparse_table<int, function<int(int, int)>> rmq;
	suffix_array(const Str &s): N(s.size()), c(N), s(s){
		p = sort_cyclic_shifts(s + "$");
		p.erase(p.begin());
		for(int i = 0; i < N; ++ i) c[p[i]] = i;
		l = get_lcp(p);
		rmq = sparse_table<int, function<int(int, int)>>(l);
	}
	vi sort_cyclic_shifts(const Str &s){
		int n = s.size();
		vi p(n), c(n), cnt(max(lim, n));
		for(auto x: s) ++ cnt[x];
		for(int i = 1; i < lim; ++ i) cnt[i] += cnt[i - 1];
		for(int i = 0; i < n; ++ i) p[-- cnt[s[i]]] = i;
		int classes = 1;
		for(int i = 1; i < n; ++ i){
			if(s[p[i]] != s[p[i - 1]]) classes ++;
			c[p[i]] = classes - 1;
		}
		vi pn(n), cn(n);
		for(int h = 0; (1 << h) < n; ++ h){
			for(int i = 0; i < n; ++ i){
				pn[i] = p[i] - (1 << h);
				if(pn[i] < 0) pn[i] += n;
			}
			fill(cnt.begin(), cnt.begin() + classes, 0);
			for(auto x: pn) ++ cnt[c[x]];
			for(int i = 1; i < classes; ++ i) cnt[i] += cnt[i - 1];
			for(int i = n - 1; i >= 0; -- i) p[-- cnt[c[pn[i]]]] = pn[i];
			cn[p[0]] = 0, classes = 1;
			for(int i = 1; i < n; ++ i){
				if(c[p[i]] != c[p[i - 1]] || c[(p[i] + (1 << h)) % n] != c[(p[i - 1] + (1 << h)) % n]){
					++ classes;
				}
				cn[p[i]] = classes - 1;
			}
			c.swap(cn);
		}
		return p;
	}
	vi get_lcp(const vi &p){
		int n = sz(s);
		vi rank(n);
		for(int i = 0; i < n; ++ i) rank[p[i]] = i;
		int k = 0;
		vi l(n - 1);
		for(int i = 0; i < n; ++ i){
			if(rank[i] == n - 1){
				k = 0;
				continue;
			}
			int j = p[rank[i] + 1];
			while(i + k < n && j + k < n && s[i + k] == s[j + k]) ++ k;
			l[rank[i]] = k;
			if(k) -- k;
		}
		return l;
	}
	int lcp(int i, int j){
		return rmq.query(min(c[i], c[j]), max(c[i], c[j]));
	}
};

// 156485479_5_4
// Z Function / for each position i > 0, returns the length of the longest prefix which is also a prefix starting at i
// O(n)
template<class Str>
vi z_function(const Str &s){
	int n = sz(s);
	vi z(n);
	for(int i = 1, l = 0, r = 1; i < n; ++ i){
		if(i < r) z[i] = min(r - i, z[i - l]);
		while(i + z[i] < n && s[z[i]] == s[i + z[i]]) ++ z[i];
		if(i + z[i] > r) l = i, r = i + z[i];
	}
	return z;
}

// 156485479_5_5
// Aho Corasic Algorithm ( construct an automaton )
// O(W) preprocessing, O(L) per query
template<int C>
struct aho_corasic{
	struct node{
		int par, link = -1, elink = -1;
		char cpar;
		vector<int> next, go;
		bool isleaf = false;
		int ind;
		node(int par = -1, char pch = '$'): par(par), cpar(pch), next(C, -1), go(C, -1){}
	};
	vector<node> arr;
	function<int(char)> trans;
	aho_corasic(function<int(char)> trans = [](char c){return c < 'Z' ? c - 'A' : c - 'a';}):
		arr(1), trans(trans){}
	void insert(int ind, const string &s){
		int u = 0;
		for(auto &ch: s){
			int c = trans(ch);
			if(arr[u].next[c] == -1){
				arr[u].next[c] = arr.size();
				arr.emplace_back(u, ch);
			}
			u = arr[u].next[c];
		}
		arr[u].isleaf = true;
		arr[u].ind = ind;
	}
	int get_link(int u){
		if(arr[u].link == -1){
			if(!u || !arr[u].par) arr[u].link = 0;
			else arr[u].link = go(get_link(arr[u].par), arr[u].cpar);
		}
		return arr[u].link;
	}
	int get_elink(int u){
		if(arr[u].elink == -1){
			if(!u || !get_link(u)) arr[u].elink = 0;
			else if(arr[get_link(u)].isleaf) arr[u].elink = get_link(u);
			else arr[u].elink = get_elink(get_link(u));
		}
		return arr[u].elink;
	}
	int go(int u, char ch){
		int c = trans(ch);
		if(arr[u].go[c] == -1){
			if(arr[u].next[c] != -1) arr[u].go[c] = arr[u].next[c];
			else arr[u].go[c] = u == 0 ? 0 : go(get_link(u), ch);
		}
		return arr[u].go[c];
	}
	void print(int u, string s = ""){
		cout << "Node " << u << ": par = " << arr[u].par << ", cpar = " << arr[u].cpar << ", string: " << s << "\n";
		for(int i = 0; i < C; ++ i){
			if(arr[u].next[i] != -1){
				cout << u << " => ";
				print(arr[u].next[i], s + string(1, i + 'a'));
			}
		}
	}
};

// 156485479_5_6
// Prefix Function / Prefix Automaton
// O(N) each
template<class Str>
vi prefix_function(const Str &s){
	int n = sz(s);
	vi p(n);
	for(int i = 1; i < n; ++ i){
		int j = p[i - 1];
		while(j > 0 && s[i] != s[j]) j = p[j - 1];
		if(s[i] == s[j]) ++ j;
		p[i] = j;
	}
	return p;
}
template<class Str, class UO = function<char(int)>, int lim = 26>
pair<vi, vector<vi>> prefix_automaton(const Str &s, UO trans = [](int c){return c + 'a';}){
	vi p = prefix_function(s);
	int n = sz(s);
	vector<vi> aut(n, vi(lim));
	for(int i = 0; i < n; ++ i){
		for(int c = 0; c < lim; ++ c){
			if(i > 0 && trans(c) != s[i]) aut[i][c] = aut[p[i - 1]][c];
			else aut[i][c] = i + (trans(c) == s[i]);
		}
	}
	return {p, aut};
}

// 156485479_5_7
// Polynomial Hash
// O(n) processing, O(log n) for lcp, O(n) for search, O(1) for query
template<class Str>
struct polyhash: vector<vl>{
	const int lim;
	const ll base, mod;
	vl p;
	polyhash(int lim, ll mod): lim(lim), p(lim, 1), mod(mod), base(rngll() % (ll)(0.4 * mod) + 0.3 * mod){
		for(int i = 1; i < lim; ++ i) p[i] = p[i - 1] * base % mod;
	}
	void insert(const Str &s){
		this->emplace_back(sz(s) + 1);
		for(int i = 0; i < sz(s); ++ i) this->back()[i + 1] = (this->back()[i] * base + s[i]) % mod;
	}
	template<class Char>
	void extend(Char c, int i = 0){
		(*this)[i].push_back(((*this)[i].back() * base + c) % mod);
	}
	ll query(int ql, int qr, int i = 0){
		return ((*this)[i][qr] - (*this)[i][ql] * p[qr - ql] % mod + mod) % mod;
	}
	int lcp(int i, int j, int posi = 0, int posj = 0){ // returns the length
		int low = 0, high = min(sz((*this)[i]) - posi, sz((*this)[j]) - posj);
		while(high - low > 1){
			int mid = low + high >> 1;
			query(posi, posi + mid, i) == query(posj, posj + mid, j) ? low = mid : high = mid;
		}
		return low;
	}
	vi search(const Str &s, bool FIND_ALL = true, int i = 0){
		int len = sz(s);
		ll v = 0;
		for(auto c: s) v = (v * base + c) % mod;
		vi res;
		for(int j = 0; j + len < sz((*this)[i]); ++ j) if(v == query(j, j + len, i)){
			res.push_back(j);
			if(!FIND_ALL) break;
		}
		return res;
	}
};
template<class Str>
struct double_polyhash{
	pair<polyhash<Str>, polyhash<Str>> h;
	double_polyhash(int N, ll mod): h{polyhash<Str>(N, mod), polyhash<Str>(N, mod)}{ }
	void insert(const Str &s){
		h.first.insert(s), h.second.insert(s);
	}
	template<class Char>
	void extend(Char c, int i = 0){
		h.first.extend(c, i), h.second.extend(c, i);
	}
	pll query(int ql, int qr, int i = 0){
		return {h.first.query(ql, qr, i), h.second.query(ql, qr, i)};
	}
	int lcp(int i, int j, int posi = 0, int posj = 0){ // returns the length
		int low = 0, high = min(sz(h.first[i]) - posi, sz(h.first[j]) - posj);
		while(high - low > 1){
			int mid = low + high >> 1;
			query(posi, posi + mid, i) == query(posj, posj + mid, j) ? low = mid : high = mid;
		}
		return low;
	}
	vi search(const Str &s, bool FIND_ALL = true, int i = 0){
		int len = sz(s);
		pll v;
		for(auto c: s) v = {(v.first * h.first.base + c) % h.first.mod, (v.second * h.second.base + c) % h.second.mod};
		vi res;
		for(int j = 0; j + len < sz(h.first[i]); ++ j) if(v == query(j, j + len, i)){
			res.push_back(j);
			if(!FIND_ALL) break;
		}
		return res;
	}
};

// 156485479_6_1
// 2D Geometry Classes
template<class T = ll> struct point{
	T x, y;
	template<class U> point(const point<U> &otr): x(otr.x), y(otr.y){ }
	template<class U, class V> point(const pair<U, V> &p): x(p.first), y(p.second){ }
	template<class U = T, class V = T> point(U x = 0, V y = 0): x(x), y(y){ }
	template<class U> explicit operator point<U>() const{ return point<U>(static_cast<U>(x), static_cast<U>(y)); }
	T operator*(const point &otr) const{ return x * otr.x + y * otr.y; }
	T operator^(const point &otr) const{ return x * otr.y - y * otr.x; }
	point operator+(const point &otr) const{ return point(x + otr.x, y + otr.y); }
	point &operator+=(const point &otr){ return *this = *this + otr; }
	point operator-(const point &otr) const{ return point(x - otr.x, y - otr.y); }
	point &operator-=(const point &otr){ return *this = *this - otr; }
	point operator*(const T &c) const{ return point(x * c, y * c); }
	point &operator*=(const T &c) { return *this = *this * c; }
	point operator/(const T &c) const{ return point(x / c, y / c); }
	point &operator/=(const T &c) { return *this = *this / c; }
	point operator-() const{ return point(-x, -y); }
	bool operator<(const point &otr) const{ return tie(x, y) < tie(otr.x, otr.y); }
	bool operator>(const point &otr) const{ return tie(x, y) > tie(otr.x, otr.y); }
	bool operator<=(const point &otr) const{ return tie(x, y) <= tie(otr.x, otr.y); }
	bool operator>=(const point &otr) const{ return tie(x, y) >= tie(otr.x, otr.y); }
	bool operator==(const point &otr) const{ return tie(x, y) == tie(otr.x, otr.y); }
	bool operator!=(const point &otr) const{ return tie(x, y) != tie(otr.x, otr.y); }
	double norm() const{ return sqrt(x * x + y * y); }
	T sqnorm() const{ return x * x + y * y; }
	double arg() const{ return atan2(y, x); } // [-pi, pi]
	point<double> unit() const{ return point<double>(x, y) / norm(); }
	point perp() const{ return point(-y, x); }
	point<double> normal() const{ return perp().unit(); }
	point<double> rotate(const double &theta) const{ return point<double>(x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)); }
	point reflect_x() const{ return point(x, -y); }
	point reflect_y() const{ return point(-x, y); }
};
template<class T> point<T> operator*(const T &c, const point<T> &p){ return point<T>(c * p.x, c * p.y); }
template<class T> istream &operator>>(istream &in, point<T> &p){ return in >> p.x >> p.y; }
template<class T> ostream &operator<<(ostream &out, const point<T> &p){ return out << pair<T, T>(p.x, p.y); }
template<class T> double distance(const point<T> &p, const point<T> &q){ return (p - q).norm(); }
template<class T> T squared_distance(const point<T> &p, const point<T> &q){ return (p - q).sqnorm(); }
template<class T, class U, class V> T ori(const point<T> &p, const point<U> &q, const point<V> &r){ return (q - p) ^ (r - p); }
template<class T> T doubled_signed_area(const vector<point<T>> &arr){
	T s = arr.back() ^ arr.front();
	for(int i = 1; i < sz(arr); ++ i) s += arr[i - 1] ^ arr[i];
	return s;
}
template<class T = ll> struct line{
	point<T> p, d; // p + d*t
	template<class U = T, class V = T> line(point<U> p = {0, 0}, point<V> q = {0, 0}, bool Two_Points = true): p(p), d(Two_Points ? q - p : q){ }
	template<class U> line(point<U> d): p(), d(static_cast<point<T>>(d)){ }
	line(T a, T b, T c): p(a ? -c / a : 0, !a && b ? -c / b : 0), d(-b, a){ }
	template<class U> explicit operator line<U>() const{ return line<U>(point<U>(p), point<U>(d), false); }
	point<T> q() const{ return p + d; }
	bool degen() const{ return d == point<T>(); }
	tuple<T, T, T> coef(){ return {d.y, -d.x, d.perp() * p}; } // d.y (X - p.x) - d.x (Y - p.y) = 0
};
template<class T> bool parallel(const line<T> &L, const line<T> &M){ return !(L.d ^ M.d); }
template<class T> bool on_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	return (p - L.p) ^ L.d == 0;
}
template<class T> bool on_ray(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return !(a ^ b) && a * L.d <= 0;
}
template<class T> bool on_segment(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return !(a ^ b) && a * b <= 0;
}
template<class T> double distance_to_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return distance(p, L.p);
	return abs((p - L.p) ^ L.d) / L.d.norm();
}
template<class T> double distance_to_ray(const point<T> &p, const line<T> &L){
	if((p - L.p) * L.d <= 0) return distance(p, L.p);
	return distance_to_line(p, L);
}
template<class T> double distance_to_segment(const point<T> &p, const line<T> &L){
	if((p - L.p) * L.d <= 0) return distance(p, L.p);
	if((p - L.q()) * L.d >= 0) return distance(p, L.q());
	return distance_to_line(p, L);
}
template<class T> point<double> projection(const point<T> &p, const line<T> &L){ return static_cast<point<double>>(L.p) + (L.degen() ? point<double>() : (p - L.p) * L.d / L.d.norm() * static_cast<point<double>>(L.d)); }
template<class T> point<double> reflection(const point<T> &p, const line<T> &L){ return 2.0 * projection(p, L) - static_cast<point<double>>(p); }
template<class T> point<double> closest_point_on_segment(const point<T> &p, const line<T> &L){ return (p - L.p) * L.d <= 0 ? static_cast<point<double>>(L.p) : ((p - L.q()) * L.d >= 0 ? static_cast<point<double>>(L.q()) : projection(p, L)); }
template<int TYPE> struct EndpointChecker{ };
// For rays
template<> struct EndpointChecker<0>{ template<class T> bool operator()(const T& a, const T& b) const{ return true; } }; // For ray
// For closed end
template<> struct EndpointChecker<1>{ template<class T> bool operator()(const T& a, const T& b) const{ return a <= b; } }; // For closed end
// For open end
template<> struct EndpointChecker<2>{ template<class T> bool operator()(const T& a, const T& b) const{ return a < b; } }; // For open end
// Assumes parallel lines do not intersect
template<int LA, int LB, int RA, int RB, class T> pair<bool, point<double>> intersect_no_parallel_overlap(const line<T> &L, const line<T> &M){
	auto s = L.d ^ M.d;
	if(!s) return {false, point<double>()};
	auto ls = (M.p - L.p) ^ M.d, rs = (M.p - L.p) ^ L.d;
	if(s < 0) s = -s, ls = -ls, rs = -rs;
	bool intersect = EndpointChecker<LA>()(decltype(ls)(0), ls) && EndpointChecker<LB>()(ls, s) && EndpointChecker<RA>()(decltype(rs)(0), rs) && EndpointChecker<RB>()(rs, s);
	return {intersect, static_cast<point<double>>(L.p) + 1.0 * ls / s * static_cast<point<double>>(L.d)};
}
// Assumes parallel lines do not intersect
template<class T> pair<bool, point<double>> intersect_closed_segments_no_parallel_overlap(const line<T> &L, const line<T> &M) {
	return intersect_no_parallel_overlap<1, 1, 1, 1>(L, M);
}
// Assumes nothing
template<class T> pair<bool, line<double>> intersect_closed_segments(const line<T> &L, const line<T> &M){
	auto s = L.d ^ M.d, ls = (M.p - L.p) ^ M.d;
	if(!s){
		if(ls) return {false, line<double>()};
		auto Lp = L.p, Lq = L.q(), Mp = M.p, Mq = M.q();
		if(Lp > Lq) swap(Lp, Lq);
		if(Mp > Mq) swap(Mp, Mq);
		line<double> res(max(Lp, Mp), min(Lq, Mq));
		return {res.d >= point<double>(), res};
	}
	auto rs = (M.p - L.p) ^ L.d;
	if(s < 0) s = -s, ls = -ls, rs = -rs;
	bool intersect = 0 <= ls && ls <= s && 0 <= rs && rs <= s;
	return {intersect, line<double>(static_cast<point<double>>(L.p) + 1.0 * ls / s * static_cast<point<double>>(L.d), point<double>())};
}
template<class T> double distance_between_rays(const line<T> &L, const line<T> &M){
	if(parallel(L, M)){
		if(L.d * M.d >= 0 || (M.p - L.p) * M.d <= 0) return distance_to_line(L.p, M);
		else return distance(L.p, M.p);
	}
	else{
		if(intersect_no_parallel_overlap<1, 0, 1, 0, ll>(L, M).first) return 0;
		else return min(distance_to_ray(L.p, M), distance_to_ray(M.p, L));
	}
}
template<class T> double distance_between_segments(const line<T> &L, const line<T> &M){
	if(intersect_closed_segments(L, M).first) return 0;
	return min({distance_to_segment(L.p, M), distance_to_segment(L.q(), M), distance_to_segment(M.p, L), distance_to_segment(M.q(), L)});
}
template<class P> struct compare_by_angle{
	const P origin;
	compare_by_angle(const P &origin = P()): origin(origin){ }
	bool operator()(const P &p, const P &q) const{ return ori(origin, p, q) > 0; }
};
template<class It, class P> void sort_by_angle(It first, It last, const P &origin){
	first = partition(first, last, [&origin](const decltype(*first) &point){ return point == origin; });
	auto pivot = partition(first, last, [&origin](const decltype(*first) &point) { return point > origin; });
	compare_by_angle<P> cmp(origin);
	sort(first, pivot, cmp), sort(pivot, last, cmp);
}

// 156485479_6_2
// Convex Hull and Minkowski Sum
// O(n log n) construction, O(n) if sorted.
template<class Polygon>
struct convex_hull: pair<Polygon, Polygon>{ // (Lower, Upper) type {0: both, 1: lower, 2: upper}
	int type;
	convex_hull(Polygon arr = Polygon(), int type = 0, bool is_sorted = false): type(type){
		if(!is_sorted) sort(all(arr)), arr.resize(unique(all(arr)) - arr.begin());
#define ADDP(C, cmp) while(sz(C) > 1 && ori(C[sz(C) - 2], p, C.back()) cmp 0) C.pop_back(); C.push_back(p);
		for(auto &p: arr){
			if(type < 2){ ADDP(this->first, >=) }
			if(!(type & 1)){ ADDP(this->second, <=) }
		}
		reverse(all(this->second));
	}
	Polygon get_hull() const{
		if(type) return type == 1 ? this->first : this->second;
		if(sz(this->first) <= 1) return this->first;
		Polygon res(this->first);
		res.insert(res.end(), ++ this->second.begin(), -- this->second.end());
		return res;
	}
	int min_element(const class Polygon::value_type &p) const{
		assert(p.y >= 0 && !this->first.empty());
		int low = 0, high = sz(this->first);
		while(high - low > 2){
			int mid1 = (2 * low + high) / 3, mid2 = (low + 2 * high) / 3;
			p * this->first[mid1] >= p * this->first[mid2] ? low = mid1 : high = mid2;
		}
		int res = low;
		for(int i = low + 1; i < high; i ++) if(p * this->first[res] > p * this->first[i]) res = i;
		return res;
	}
	int max_element(const class Polygon::value_type &p) const{
		assert(p.y >= 0 && !this->second.empty());
		int low = 0, high = sz(this->second);
		while(high - low > 2){
			int mid1 = (2 * low + high) / 3, mid2 = (low + 2 * high) / 3;
			p * this->second[mid1] <= p * this->second[mid2] ? low = mid1 : high = mid2;
		}
		int res = low;
		for(int i = low + 1; i < high; ++ i) if(p * this->second[res] < p * this->second[i]) res = i;
		return res;
	}
	Polygon linearize() const{
		if(type == 1) return this->first;
		if(type == 2){ Polygon res(this->second); reverse(all(res)); return res; }
		if(sz(this->first) <= 1) return this->first;
		Polygon res;
		res.reserve(sz(this->first) + sz(this->second));
		merge(all(this->first), ++ this->second.rbegin(), -- this->second.rend(), back_inserter(res));
		return res;
	}
	convex_hull operator^(const convex_hull &otr) const{
		Polygon temp, A = linearize(), B = otr.linearize();
		temp.reserve(sz(A) + sz(B));
		merge(all(A), all(B), back_inserter(temp));
		temp.resize(unique(all(temp)) - temp.begin());
		return {temp, type, true};
	}
	pair<Polygon, Polygon> get_boundary() const{
		Polygon L(this->first), R(this->second);
		for(int i = sz(L) - 1; i > 0; -- i) L[i] -= L[i - 1];
		for(int i = sz(R) - 1; i > 0; -- i) R[i] -= R[i - 1];
		return {L, R};
	}
	convex_hull operator+(const convex_hull &otr) const{
		assert(type == otr.type);
		convex_hull res(Polygon(), type);
		pair<Polygon, Polygon> A(this->get_boundary()), B(otr.get_boundary());
		compare_by_angle<class Polygon::value_type> cmp;
#define PROCESS(COND, X) \
if(COND && !A.X.empty() && !B.X.empty()){ \
	res.X.reserve(sz(A.X) + sz(B.X)); \
	res.X.push_back(A.X.front() + B.X.front()); \
	merge(A.X.begin() + 1, A.X.end(), B.X.begin() + 1, B.X.end(), back_inserter(res.X), cmp); \
	for(int i = 1; i < sz(res.X); ++ i) res.X[i] += res.X[i - 1]; \
}
		PROCESS(type < 2, first)
		PROCESS(!(type & 1), second)
		return res;
	}
};

// 156485479_7_1
// Custom Hash Function for unordered_set and unordered map
struct custom_hash{
	static uint64_t splitmix64(uint64_t x){
		// http://xorshift.di.unimi.it/splitmix64.c
		x += 0x9e3779b97f4a7c15;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
		x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
		return x ^ (x >> 31);
	}
	size_t operator()(uint64_t x) const {
		static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
		return splitmix64(x + FIXED_RANDOM);
	}
};
// KACTL Hash Function
# define M_PI 3.141592653589793238462643383279502884L
const int RANDOM = rnd();
struct custom_hash{ // To use most bits rather than just the lowest ones:
	static const uint64_t C = ll(2e18 * M_PI) + 71; // large odd number
	ll operator()(ll x) const { return __builtin_bswap64((x^RANDOM)*C); }
};

/*
Speed test results
set                                    1e6: 670ms | 1e7: 10155ms
unordered_set                          1e6: 296ms | 1e7: 4320ms
unordered_set with custom hash         1e6: 358ms | 1e7: 4851ms
unordered_set with custom hash(narut)  1e6: 389ms | 1e7: 4850ms
unordered_set with custom hash(pajen)  1e6: 436ms | 1e7: 5022ms

map                                    1e6: 592ms | 1e7: 10420ms
unordered_map                          1e6: 373ms | 1e7: 4742ms
unordered_map with custom hash         1e6: 389ms | 1e7: 4913ms
unordered_map with custom hash(narut)  1e6: 327ms | 1e7: 4960ms
unordered_map with custom hash(pajen)  1e6: 389ms | 1e7: 4789ms

map           | 1e6: 576ms 31560KB | 5e6: 4757ms 156552KB | 1e7: 10498ms 313280KB
unodered_map  | 1e6: 327ms 32220KB | 5e6: 2121ms 147132KB | 1e7: 4835ms  295068KB
cc_hash_table | 1e6: 249ms 31916KB | 5e6: 2011ms 197140KB | 1e7: 4383ms  394588KB
gp_hash_table | 1e6: 109ms 36720KB | 5e6: 686ms  295516KB | 1e7: ????    MLE
*/

// 156485479_7_2
// Bump Allocator
static char BUFF[450 << 20];
void *operator new(size_t s){
	static size_t i = sizeof BUFF;
	assert(s < i);
	return (void *)&BUFF[i -= s];
}
void operator delete(void *){}