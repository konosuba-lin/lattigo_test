package ringqp

import (
	"github.com/tuneinsight/lattigo/v4/ring"
	"github.com/tuneinsight/lattigo/v4/utils/sampling"
)

// UniformSampler is a type for sampling polynomials in Ring.
type UniformSampler struct {
	samplerQ, samplerP *ring.UniformSampler
}

// NewUniformSampler instantiates a new UniformSampler from a given PRNG.
func NewUniformSampler(prng sampling.PRNG, r Ring) (s UniformSampler) {
	if r.RingQ != nil {
		s.samplerQ = ring.NewUniformSampler(prng, r.RingQ)
	}

	if r.RingP != nil {
		s.samplerP = ring.NewUniformSampler(prng, r.RingP)
	}

	return s
}

// AtLevel returns a shallow copy of the target sampler that operates at the specified levels.
func (s UniformSampler) AtLevel(levelQ, levelP int) UniformSampler {

	var samplerQ, samplerP *ring.UniformSampler

	if levelQ > -1 {
		samplerQ = s.samplerQ.AtLevel(levelQ)
	}

	if levelP > -1 {
		samplerP = s.samplerP.AtLevel(levelP)
	}

	return UniformSampler{
		samplerQ: samplerQ,
		samplerP: samplerP,
	}
}

// Read samples a new polynomial in Ring and stores it into p.
func (s UniformSampler) Read(p *Poly) {
	if p.Q != nil && s.samplerQ != nil {
		s.samplerQ.Read(p.Q)
	}

	if p.P != nil && s.samplerP != nil {
		s.samplerP.Read(p.P)
	}
}

// ReadNew samples a new polynomial in Ring and returns it.
func (s UniformSampler) ReadNew() *Poly {

	var Q, P *ring.Poly
	if s.samplerQ != nil {
		Q = s.samplerQ.ReadNew()
	}

	if s.samplerP != nil {
		P = s.samplerP.ReadNew()
	}

	return &Poly{Q, P}
}

func (s UniformSampler) WithPRNG(prng sampling.PRNG) UniformSampler {
	sp := UniformSampler{samplerQ: s.samplerQ.WithPRNG(prng)}
	if s.samplerP != nil {
		sp.samplerP = s.samplerP.WithPRNG(prng)
	}
	return sp
}