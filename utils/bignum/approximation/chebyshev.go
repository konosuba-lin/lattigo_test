// Package approximation provides methods to approximate functions with polynomials in a given interval.
package approximation

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v4/utils/bignum"
	"github.com/tuneinsight/lattigo/v4/utils/bignum/polynomial"
)

// Chebyshev computes a Chebyshev approximation of the input function, for the range [-a, b] of degree degree.
// function.(type) can be either :
// - func(bignum.Complex128)bignum.Complex128
// - func(float64)float64
// - func(*big.Float)*big.Float
// - func(*bignum.Complex)*bignum.Complex
// The reference precision is taken from the values stored in the Interval struct.
func Chebyshev(f func(*bignum.Complex) *bignum.Complex, interval polynomial.Interval, degree int) (pol *polynomial.Polynomial) {

	nodes := chebyshevNodes(degree+1, interval)

	fi := make([]*bignum.Complex, len(nodes))

	x := bignum.NewComplex()
	x.SetPrec(interval.A.Prec())

	for i := range nodes {
		x[0].Set(nodes[i])
		fi[i] = f(x)
	}

	return polynomial.NewPolynomial(polynomial.Chebyshev, chebyCoeffs(nodes, fi, interval), &interval)
}

func chebyshevNodes(n int, interval polynomial.Interval) (u []*big.Float) {

	prec := interval.A.Prec()

	u = make([]*big.Float, n)

	half := new(big.Float).SetPrec(prec).SetFloat64(0.5)

	x := new(big.Float).Add(&interval.A, &interval.B)
	x.Mul(x, half)
	y := new(big.Float).Sub(&interval.B, &interval.A)
	y.Mul(y, half)

	PiOverN := bignum.Pi(prec)
	PiOverN.Quo(PiOverN, new(big.Float).SetInt64(int64(n)))

	for k := 1; k < n+1; k++ {
		up := new(big.Float).SetPrec(prec).SetFloat64(float64(k) - 0.5)
		up.Mul(up, PiOverN)
		up = bignum.Cos(up)
		up.Mul(up, y)
		up.Add(up, x)
		u[k-1] = up
	}

	return
}

func chebyCoeffs(nodes []*big.Float, fi []*bignum.Complex, interval polynomial.Interval) (coeffs []*bignum.Complex) {

	prec := interval.A.Prec()

	n := len(nodes)

	coeffs = make([]*bignum.Complex, n)
	for i := range coeffs {
		coeffs[i] = bignum.NewComplex().SetPrec(prec)
	}

	u := bignum.NewComplex().SetPrec(prec)

	mul := bignum.NewComplexMultiplier()

	tmp := bignum.NewComplex().SetPrec(prec)

	two := new(big.Float).SetPrec(prec).SetInt64(2)

	minusab := new(big.Float).Set(&interval.A)
	minusab.Neg(minusab)
	minusab.Sub(minusab, &interval.B)

	bminusa := new(big.Float).Set(&interval.B)
	bminusa.Sub(bminusa, &interval.A)

	Tnext := bignum.NewComplex().SetPrec(prec)

	for i := 0; i < n; i++ {

		u[0].Mul(nodes[i], two)
		u[0].Sub(u[0], minusab)
		u[0].Quo(u[0], bminusa)

		Tprev := bignum.NewComplex().SetPrec(prec)
		Tprev[0].SetFloat64(1)

		T := u.Clone()

		for j := 0; j < n; j++ {

			mul.Mul(fi[i], Tprev, tmp)
			coeffs[j].Add(coeffs[j], tmp)

			mul.Mul(u, T, Tnext)
			Tnext[0].Mul(Tnext[0], two)
			Tnext[1].Mul(Tnext[1], two)
			Tnext.Sub(Tnext, Tprev)

			Tprev.Set(T)
			T.Set(Tnext)
		}
	}

	NHalf := new(big.Float).SetInt64(int64(n))

	coeffs[0][0].Quo(coeffs[0][0], NHalf)
	coeffs[0][1].Quo(coeffs[0][1], NHalf)

	NHalf.Quo(NHalf, two)

	for i := 1; i < n; i++ {
		coeffs[i][0].Quo(coeffs[i][0], NHalf)
		coeffs[i][1].Quo(coeffs[i][1], NHalf)
	}

	return
}