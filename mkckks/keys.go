package mkckks

import "github.com/ldsec/lattigo/v2/mkrlwe"

// NewKeyGenerator creates a rlwe.KeyGenerator instance from the CKKS parameters.
func NewKeyGenerator(params Parameters) *mkrlwe.KeyGenerator {
	return mkrlwe.NewKeyGenerator(params.Parameters)
}
