package hebase

import (
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

type TestParametersLiteral struct {
	BaseTwoDecomposition int
	rlwe.ParametersLiteral
}

var (
	logN = 10
	qi   = []uint64{0x200000440001, 0x7fff80001, 0x800280001, 0x7ffd80001, 0x7ffc80001}
	pj   = []uint64{0x3ffffffb80001, 0x4000000800001}

	testParamsLiteral = []TestParametersLiteral{
		// RNS decomposition, no Pw2 decomposition
		{
			BaseTwoDecomposition: 0,

			ParametersLiteral: rlwe.ParametersLiteral{
				LogN:    logN,
				Q:       qi,
				P:       pj,
				NTTFlag: true,
			},
		},
		// RNS decomposition, Pw2 decomposition
		{
			BaseTwoDecomposition: 16,

			ParametersLiteral: rlwe.ParametersLiteral{
				LogN:    logN,
				Q:       qi,
				P:       pj[:1],
				NTTFlag: true,
			},
		},
		// No RNS decomposition, Pw2 decomposition
		{
			BaseTwoDecomposition: 1,

			ParametersLiteral: rlwe.ParametersLiteral{
				LogN:    logN,
				Q:       qi,
				P:       nil,
				NTTFlag: true,
			},
		},
	}
)