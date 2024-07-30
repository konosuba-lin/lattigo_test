// Package main implements an example showcasing the basics of the bootstrapping for the CKKS scheme.
// The CKKS bootstrapping is a circuit that homomorphically re-encrypts a ciphertext at level zero to a ciphertext at a higher level, enabling further computations.
// Note that, unlike the BGV or BFV bootstrapping, the CKKS bootstrapping does not reduce the error in the ciphertext, but only enables further computations.
// Use the flag -short to run the examples fast but with insecure parameters.
package main

import (
	"flag"
	"fmt"
	"math"
	//"math/cmplx"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/utils"
	"github.com/tuneinsight/lattigo/v4/ring"
	"github.com/pkg/profile"
	//"os"
	//"strconv"
)

var flagShort = flag.Bool("short", false, "run the example with a smaller and insecure ring degree.")

func main() {
	var mul_cnt[11] int
	mul_cnt[0] = ring.MUL_COUNT
	flag.Parse()
	defer profile.Start(profile.MemProfile).Stop()
	/*
	// First we define the residual CKKS parameters. This is only a template that will be given
	// to the constructor along with the specificities of the bootstrapping circuit we choose, to
	// enable it to create the appropriate ckks.ParametersLiteral that enable the evaluation of the
	// bootstrapping circuit on top of the residual moduli that we defined.
	//custom 0
	ckksParamsResidualLit := ckks.ParametersLiteral{
		LogN:     16,                                                // Log2 of the ringdegree
		LogSlots: 15,                                                // Log2 of the number of slots
		LogQ:     []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, // Log2 of the ciphertext prime moduli
		LogP:     []int{61, 61, 61, 61},                             // Log2 of the key-switch auxiliary prime moduli
		LogScale: 40,                                                // Log2 of the scale
		H:        192,                                               // Hamming weight of the secret
	}

	// Note that with H=192 and LogN=16, parameters are at least 128-bit if LogQP <= 1550.
	// Our default parameters have an expected logQP of 55 + 10*40 + 4*61 = 699, meaning
	// that the depth of the bootstrapping shouldn't be larger than 1550-699 = 851.

	// For this first example, we do not specify any optional field of the bootstrapping
	// Thus we expect the bootstrapping to give a precision of 27.25 bits with H=192 (and 23.8 with H=N/2)
	// if the plaintext values are uniformly distributed in [-1, 1] for both the real and imaginary part.
	// See `/ckks/bootstrapping/parameters.go` for information about the optional fields.
	btpParametersLit := bootstrapping.ParametersLiteral{}

	// The default bootstrapping parameters consume 822 bits which is smaller than the maximum
	// allowed of 851 in our example, so the target security is easily met.
	// We can print and verify the expected bit consumption of bootstrapping parameters with:
	bits, err := btpParametersLit.BitConsumption()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Bootstrapping depth (bits): %d\n", bits)
	*/

	// Now we generate the updated ckks.ParametersLiteral that contain our residual moduli and the moduli for
	// the bootstrapping circuit, as well as the bootstrapping.Parameters that contain all the necessary information
	// of the bootstrapping circuit.
	/*
	param_type := os.Args[1]
	param_idx, e:= strconv.Atoi(os.Args[2]) 
    if e == nil { 
        fmt.Printf("%T \n %v", param_idx, param_idx) 
    } 
	paramSet := bootstrapping.DefaultParametersSparse[0]
	if (param_idx>=4) {
		fmt.Printf("Error: param_idx>4, use default param sparse_0\n")
	} else if ((param_type!="sparse") && (param_type!="dense")) {
		fmt.Printf("Error: param_type!=Sparse/Dense, use default param sparse_0\n")
	} else {
		if param_type=="sparse" {
			fmt.Printf("\n\nParameter =  sparse_%d\n",param_idx)
			paramSet = bootstrapping.DefaultParametersSparse[param_idx]
		} else {
			fmt.Printf("\n\nParameter =  dense_%d\n",param_idx)
			paramSet = bootstrapping.DefaultParametersDense[param_idx]
		}
	}
	ckksParamsResidualLit := paramSet.SchemeParams
	btpParametersLit := paramSet.BootstrappingParams
	*/
	paramSet := bootstrapping.DefaultParametersSparse[2]
	ckksParamsResidualLit := paramSet.SchemeParams
	btpParametersLit := paramSet.BootstrappingParams
	ckksParamsLit, btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParamsResidualLit, btpParametersLit)
	fmt.Printf("paramSet : %p\n", ckksParamsLit)
	if err != nil {
		panic(err)
	}

	if *flagShort {

		prevLogSlots := ckksParamsLit.LogSlots

		ckksParamsLit.LogN = 13

		// Corrects the message ratio to take into account the smaller number of slots and keep the same precision
		btpParams.EvalModParameters.LogMessageRatio += prevLogSlots - ckksParamsLit.LogN - 1

		ckksParamsLit.LogSlots = ckksParamsLit.LogN - 1
	}

	// This generate ckks.Parameters, with the NTT tables and other pre-computations from the ckks.ParametersLiteral (which is only a template).
	params, err := ckks.NewParametersFromLiteral(ckksParamsLit)
	if err != nil {
		panic(err)
	}

	// Here we print some information about the generated ckks.Parameters
	// We can notably check that the LogQP of the generated ckks.Parameters is equal to 699 + 822 = 1521.
	// Not that this value can be overestimated by one bit.
	fmt.Printf("CKKS parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%f, logQP=%d, levels=%d, scale=2^%f\n", params.LogN(), params.LogSlots(), params.HammingWeight(), btpParams.EphemeralSecretWeight, params.Sigma(), params.LogQP(), params.QCount(), math.Log2(params.DefaultScale().Float64()))

	// Scheme context and keys
	kgen := ckks.NewKeyGenerator(params)

	sk, pk := kgen.GenKeyPair()

	encoder := ckks.NewEncoder(params)
	decryptor := ckks.NewDecryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, pk)

	mul_cnt[1] = ring.MUL_COUNT
	fmt.Println()
	fmt.Println("Generating bootstrapping keys...")
	evk := bootstrapping.GenEvaluationKeys(btpParams, params, sk)
	fmt.Println("Done")
	mul_cnt[2] = ring.MUL_COUNT
	fmt.Println("Calculating bootstrapping key size...")
	Rlk_size := evk.EvaluationKey.Rlk.MarshalBinarySize()
	Rtks_size := evk.EvaluationKey.Rtks.MarshalBinarySize()
	SwkDtS_size := evk.SwkDtS.MarshalBinarySize()
	SwkStD_size := evk.SwkStD.MarshalBinarySize()
	fmt.Printf("Rlk%d\n", Rlk_size)
	fmt.Printf("Rtks:%d\n", Rtks_size)
	fmt.Printf("SwkDtS:%d\n", SwkDtS_size)
	fmt.Printf("SwkStD:%d\n", SwkStD_size)
	fmt.Printf("Total:%d\n", Rlk_size+Rtks_size+SwkDtS_size+SwkStD_size)

	var btp *bootstrapping.Bootstrapper
	if btp, err = bootstrapping.NewBootstrapper(params, btpParams, evk); err != nil {
		panic(err)
	}

	// Generate a random plaintext with values uniformly distributed in [-1, 1] for the real and imaginary part.
	valuesWant := make([]complex128, params.Slots())
	for i := range valuesWant {
		valuesWant[i] = utils.RandComplex128(-1, 1)
	}

	mul_cnt[3] = ring.MUL_COUNT
	plaintext := encoder.EncodeNew(valuesWant, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	mul_cnt[4] = ring.MUL_COUNT
	// Encrypt
	mul_cnt[5] = ring.MUL_COUNT
	ciphertext1 := encryptor.EncryptNew(plaintext)
	mul_cnt[6] = ring.MUL_COUNT
	// Decrypt, print and compare with the plaintext values
	fmt.Println()
	fmt.Println("Precision loss during encryption")
	valuesTest1 := printDebug(params, ciphertext1, valuesWant, decryptor, encoder)

	// Bootstrap the ciphertext (homomorphic re-encryption)
	// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
	// and returns a ciphertext with the max level of `ckksParamsResidualLit`.
	// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.DefaultScale()
	// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.DefaultScale()) can be used at the expense of one level.
	// If the ciphertext is is at level one or greater when given to the bootstrapper, this equalization is automatically done.
	fmt.Println()
	fmt.Println("Bootstrapping...")
	mul_cnt[7] = ring.MUL_COUNT
	ciphertext2 := btp.Bootstrap(ciphertext1)
	mul_cnt[8] = ring.MUL_COUNT
	fmt.Println("Done")

	// Decrypt, print and compare with the plaintext values
	fmt.Println()
	fmt.Println("Precision loss during bootstrapping")
	printDebug(params, ciphertext2, valuesTest1, decryptor, encoder)

	//valuesTest2 := printDebug(params, ciphertext2, valuesTest1, decryptor, encoder)

	//evaluate x^(pow(2,L-Lboot))
	/*r := math.Pow(2,float64(ciphertext2.Level()))
	fmt.Printf("Evaluate x^pow(2,%d) = x^%d\n",ciphertext2.Level(),int(r))
	slots := params.Slots()
	valuesWant2 := make([]complex128, slots)
	valuesWant3 := make([]complex128, slots)
	for i := range valuesWant2 {
		valuesWant2[i] = cmplx.Pow(valuesTest2[i], complex(r, 0))
		valuesWant3[i] = cmplx.Pow(valuesWant[i], complex(r, 0))
	}
	rlk := kgen.GenRelinearizationKey(sk, 1)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})
	monomialBasis := ckks.NewPolynomialBasis(ciphertext2, ckks.Monomial)
	mul_cnt[9] = ring.MUL_COUNT
	monomialBasis.GenPower(int(r), false, params.DefaultScale(), evaluator)
	mul_cnt[10] = ring.MUL_COUNT
	ciphertext3 := monomialBasis.Value[int(r)]
	fmt.Printf("Done\n")
	fmt.Println()
	fmt.Println("Precision loss during bootstrapped ciphertext^r\n")
	printDebug(params, ciphertext3, valuesWant2, decryptor, encoder)
	//precision loss blows up becasue x is in [-1,1] and usually |x|<1 and |x^r| are too small 
	//you can use x = [cos theta, sin theta] instead to keep |x^r|=1  

	//decrypt
	fmt.Println("Total Precision loss\n")
	encoder.Decode(decryptor.DecryptNew(ciphertext3), params.LogSlots())
	printDebug(params, ciphertext3, valuesWant3, decryptor, encoder)*/



	fmt.Printf("----------------MUL COUNT----------------\n")
	fmt.Printf("GenEvaluationKeys: %d (%d --> %d)\n",mul_cnt[2]-mul_cnt[1], mul_cnt[1], mul_cnt[2])
	fmt.Printf("EncodeNew        : %d (%d --> %d)\n",mul_cnt[4]-mul_cnt[3], mul_cnt[3], mul_cnt[4])
	fmt.Printf("EncryptNew       : %d (%d --> %d)\n",mul_cnt[6]-mul_cnt[5], mul_cnt[5], mul_cnt[6])
	fmt.Printf("Bootstrapping    : %d (%d --> %d)\n",mul_cnt[8]-mul_cnt[7], mul_cnt[7], mul_cnt[8])
	fmt.Printf("Power            : %d (%d --> %d)\n",mul_cnt[10]-mul_cnt[9], mul_cnt[9], mul_cnt[10])
}

func printDebug(params ckks.Parameters, ciphertext *rlwe.Ciphertext, valuesWant []complex128, decryptor rlwe.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))

	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale.Float64()))
	fmt.Printf("ValuesTest: %6.10f %6.10f %6.10f %6.10f...\n", valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
	fmt.Printf("ValuesWant: %6.10f %6.10f %6.10f %6.10f...\n", valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}
