package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"

	"github.com/ldsec/lattigo/v2/mkckks"
	"github.com/ldsec/lattigo/v2/mkrlwe"
)

var (
	numData    = 2000
	numFeature = 20
	numCompany = 4

	numTrain  = 512
	numIter   = 4
	batchSize = 512

	classification_label = 0
	gamma                = 3                                    // learning rate
	eta                  = [4]float64{1, 0, -0.28175, -0.43404} // weight

	c3 = -0.0015
	c1 = 0.15
	c0 = 0.5 // sigmoid(x) = c3*x^3 + c1*x + c0

	// Variables for HE setting
	slotNum   = int(math.Pow(2, 14))
	PN15QP880 = ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: 14,
		Q: []uint64{ // 50 + 38 * 20
			0x80000000080001,
			0x4000170001, 0x40002f0001, 0x3fffe80001,
			0x4000300001, 0x40003f0001, 0x3fffcf0001,
			0x4000450001, 0x3fffc10001, 0x40004a0001, 0x3fffb80001,
			0x3fffb70001, 0x4000510001, 0x3fffb20001, 0x4000540001,
			0x3fffaf0001, 0x4000560001, 0x4000590001,
			0x3fff810001, 0x40006b0001, 0x4000720001},
		P: []uint64{ // 50 * 2
			0x40000001b0001, 0x4000000270001},
		Scale: 1 << 38,
		Sigma: rlwe.DefaultSigma,
	}
)

type testParams struct {
	params mkckks.Parameters
	ringQ  *ring.Ring
	ringP  *ring.Ring
	prng   utils.PRNG
	kgen   *mkrlwe.KeyGenerator
	skSet  *mkrlwe.SecretKeySet
	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet

	encryptor *mkckks.Encryptor
	decryptor *mkckks.Decryptor
	evaluator *mkckks.Evaluator
	idset     *mkrlwe.IDSet
}

func main() {
	const filename = "./toyData.csv"
	var featureData [][]complex128
	var labelData []complex128
	featureData, labelData = readData(filename)

	// Modify labelData correspoding to the target label
	var classification_label float64 = 0 // 0, 1, 2, 3 in this case.
	for i := 0; i < numData; i++ {
		if real(labelData[i]) == classification_label {
			labelData[i] = complex(1, 0)
		} else {
			labelData[i] = complex(0, 0)
		}
	}

	// Setting for HE
	fmt.Println()
	fmt.Println("Setting Parameters...")
	ckks_params, err := ckks.NewParametersFromLiteral(PN15QP880)
	params := mkckks.NewParameters(ckks_params)

	if err != nil {
		panic(err)
	}

	var companySet []string
	for i := 0; i < numCompany; i++ {
		companySet = append(companySet, "company"+strconv.Itoa(i))
	}

	idset := mkrlwe.NewIDSet()
	for _, id := range companySet {
		idset.Add(id)
	}

	var testContext *testParams
	if testContext, err = genTestParams(params, idset); err != nil {
		panic(err)
	}

	fmt.Println("Encrypting Data and Parameters...")

	// Initialize beta_0, v_0
	zero_msg := mkckks.NewMessage(testContext.params)
	zero_vector := make([]complex128, slotNum)
	for i := 0; i < int(slotNum); i++ {
		zero_vector[i] = complex(0, 0)
	}
	zero_msg.Value = zero_vector
	id := "company0"
	beta := testContext.encryptor.EncryptMsgNew(zero_msg, testContext.pkSet.GetPublicKey(id))
	v := testContext.encryptor.EncryptMsgNew(zero_msg, testContext.pkSet.GetPublicKey(id))

	// Normalize
	fmt.Println("Normalizing data...")
	mean := make([]float64, numFeature)
	std := make([]float64, numFeature)
	for j := 0; j < numFeature; j++ {
		for i := 0; i < batchSize; i++ {
			mean[j] += real(featureData[i][j])
		}
		mean[j] /= float64(batchSize)

		for i := 0; i < batchSize; i++ {
			std[j] += math.Pow(real(featureData[i][j])-mean[j], 2)
		}
		std[j] = math.Sqrt(std[j] / float64(batchSize))
	}

	for i := 0; i < numData; i++ {
		for j := 0; j < numFeature; j++ {
			if j != 1 && j != 3 && j != 5 && j < 17 {
				featureData[i][j] = complex((real(featureData[i][j])-mean[j])/std[j], 0)
			}
		}
	}

	// Each z has 512 data (feature num: 20(or 21) < 2^5, slot num: 2^14 --> include 2^9 data)
	logslotsPerData := int(math.Ceil(math.Log2(float64(numFeature))))
	slotsPerData := int(math.Pow(2, float64(logslotsPerData)))
	fmt.Print(slotsPerData)
	var company_feature [4][5]int
	for i := 0; i < numCompany; i++ {
		for j := 0; j < 5; j++ {
			company_feature[i][j] = i*5 + j
		}
	}

	fmt.Println("Encrypting data...")
	ztmp_msg := mkckks.NewMessage(testContext.params)
	ztmp := make([]*mkckks.Ciphertext, numCompany)
	for b := 0; b < numCompany; b++ {
		ztmp_vector := make([]complex128, slotNum)
		for i := 0; i < numTrain; i++ {
			for j := 0; j < len(company_feature[b]); j++ {
				ztmp_vector[i*slotsPerData+5*b+j] = featureData[i][company_feature[b][j]]
			}
		}
		ztmp_msg.Value = ztmp_vector
		ztmp[b] = testContext.encryptor.EncryptMsgNewExpand(ztmp_msg, testContext.pkSet.GetPublicKey(id), idset)
	}

	z := ztmp[0]
	fmt.Printf("z0: ")
	z0_decrypt := testContext.decryptor.Decrypt(z, testContext.skSet)
	fmt.Println(z0_decrypt.Value[0:6])
	fmt.Println()
	fmt.Printf("z1: ")
	z1_decrypt := testContext.decryptor.Decrypt(ztmp[1], testContext.skSet)
	fmt.Println(z1_decrypt.Value[0:6])
	fmt.Println()
	fmt.Printf("z2: ")
	z2_decrypt := testContext.decryptor.Decrypt(ztmp[2], testContext.skSet)
	fmt.Println(z2_decrypt.Value[0:6])
	fmt.Println()
	fmt.Printf("z3: ")
	z3_decrypt := testContext.decryptor.Decrypt(ztmp[3], testContext.skSet)
	fmt.Println(z3_decrypt.Value[0:6])
	fmt.Println()

	for i := 1; i < numCompany; i++ {
		fmt.Print(i)
		z = testContext.evaluator.AddNew(z, ztmp[i])
	}
	fmt.Printf("z: ")
	z_decrypt := testContext.decryptor.Decrypt(z, testContext.skSet)
	fmt.Println(z_decrypt.Value[0:10])
	fmt.Println(z_decrypt.Value[200:210])

	// Generate a label vector
	label_vector := make([]complex128, slotNum)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < int(math.Pow(2, float64(5))); j++ {
			label_vector[int(math.Pow(2, float64(5)))*i+j] = labelData[i]
		}
	}

	// Generate a mask ciphertext
	mask_msg := mkckks.NewMessage(testContext.params)
	mask_vector := make([]complex128, slotNum)
	for i := 0; i < int(slotNum); i++ {
		if i%int(math.Pow(2, float64(5))) == 0 {
			mask_vector[i] = complex(1, 0)
		} else {
			mask_vector[i] = complex(0, 0)
		}
	}
	mask_msg.Value = mask_vector
	mask := testContext.encryptor.EncodeMsgNew(mask_msg)

	// Generate constant ct & pt
	const_msg := mkckks.NewMessage(testContext.params)
	const_vector := make([]complex128, slotNum)

	// const1_pt = -(gamma / data_num * (c0-y))
	for i := 0; i < int(slotNum); i++ {
		const_vector[i] = complex(-float64(gamma)/float64(batchSize)*(c0-real(label_vector[i])), 0)
	}
	const_msg.Value = const_vector
	const1_pt := testContext.encryptor.EncodeMsgNew(const_msg)

	// const2_pt = -(1 - eta) * gamma / data_num * (c0-y)
	const2_pt := make([]*ckks.Plaintext, numIter)
	for iter := 0; iter < numIter; iter++ {
		for i := 0; i < int(slotNum); i++ {
			const_vector[i] = complex(-(1-eta[iter])*float64(gamma)/float64(batchSize)*(c0-real(label_vector[i])), 0)
		}
		const_msg.Value = const_vector
		const2_pt[iter] = testContext.encryptor.EncodeMsgNew(const_msg)
	}

	// const3_pt = -(gamma / data_num * c3)
	for i := 0; i < int(slotNum); i++ {
		const_vector[i] = complex(-float64(gamma)/float64(batchSize)*c3, 0)
	}
	const_msg.Value = const_vector
	const3_pt := testContext.encryptor.EncodeMsgNew(const_msg)

	// const4_pt = -(1 - eta) * gamma / data_num * c3
	const4_pt := make([]*ckks.Plaintext, numIter)
	for iter := 0; iter < numIter; iter++ {
		for i := 0; i < int(slotNum); i++ {
			const_vector[i] = complex(-(1-eta[iter])*float64(gamma)/float64(batchSize)*c3, 0)
		}
		const_msg.Value = const_vector
		const4_pt[iter] = testContext.encryptor.EncodeMsgNew(const_msg)
	}

	// const5_ct = c1/c3
	for i := 0; i < int(slotNum); i++ {
		const_vector[i] = complex(c1/c3, 0)
	}
	const_msg.Value = const_vector
	const5_ct := testContext.encryptor.EncryptMsgNew(const_msg, testContext.pkSet.GetPublicKey(id))

	// const6_pt = (1 - eta)
	const6_pt := make([]*ckks.Plaintext, numIter)
	for iter := 0; iter < numIter; iter++ {
		for i := 0; i < int(slotNum); i++ {
			const_vector[i] = complex(1-eta[iter], 0)
		}
		const6_pt[iter] = testContext.encryptor.EncodeMsgNew(const_msg)
	}
	// const7_pt = eta
	const7_pt := make([]*ckks.Plaintext, numIter)
	for iter := 0; iter < numIter; iter++ {
		for i := 0; i < int(slotNum); i++ {
			const_vector[i] = complex(eta[iter], 0)
		}
		const7_pt[iter] = testContext.encryptor.EncodeMsgNew(const_msg)
	}

	// const8_pt = c3
	for i := 0; i < int(slotNum); i++ {
		const_vector[i] = complex(c3, 0)
	}
	const_msg.Value = const_vector
	const8_pt := testContext.encryptor.EncodeMsgNew(const_msg)

	// const9_ct = c0
	for i := 0; i < int(slotNum); i++ {
		const_vector[i] = complex(c0, 0)
	}
	const_msg.Value = const_vector
	const9_ct := testContext.encryptor.EncryptMsgNew(const_msg, testContext.pkSet.GetPublicKey(id))

	fmt.Println()
	fmt.Println("Training...")
	start := time.Now()
	for a := 0; a < numIter; a++ {
		fmt.Println(a, "-th Iteration")
		//////////////////////////// depth 1 //////////////////////////////////
		// M_j = Z_j * V_j
		m := testContext.evaluator.MulRelinNew(z, v, testContext.rlkSet)

		// z1 = -(gamma / data_num * (c0-y)) * Z_j
		z1 := testContext.evaluator.MulPtxtNew(z, const1_pt)

		// z2 = -(1 - eta) * gamma / data_num * (c0-y) * Z_j
		z2 := testContext.evaluator.MulPtxtNew(z, const2_pt[a])

		// z3 = -(gamma / data_num * c3) * Z_j
		z3 := testContext.evaluator.MulPtxtNew(z, const3_pt)

		// z4 = -(1 - eta) * gamma / data_num * c3 * Z_j
		z4 := testContext.evaluator.MulPtxtNew(z, const4_pt[a])

		///////////////////////////// depth 2 //////////////////////////////////
		// M = \sum_j SumColVec(M_j)
		m = SumColVec(testContext.evaluator, testContext.rlkSet, testContext.rtkSet, m, mask, int(math.Pow(2, 9.0)), int(math.Pow(2, 5.0)))

		///////////////////////////// depth 3 //////////////////////////////////
		// M'' = M * M + c1/c3
		m2 := testContext.evaluator.MulRelinNew(m, m, testContext.rlkSet)
		m2 = testContext.evaluator.AddNew(m2, const5_ct)

		// M' = M * Z3_j
		m1 := testContext.evaluator.MulRelinNew(m, z3, testContext.rlkSet)

		// S_j = Z4_j * M
		s := testContext.evaluator.MulRelinNew(z4, m, testContext.rlkSet)

		///////////////////////////// depth 4 //////////////////////////////////
		// G_j = M' * M'' + Z1_j
		g := testContext.evaluator.MulRelinNew(m1, m2, testContext.rlkSet)
		g = testContext.evaluator.AddNew(g, z1)

		// W+_j = V_j + SumRowVec(G_j)

		for i := 1; i < 512; i *= 2 {
			g_rot := testContext.evaluator.RotateNew(g, 32*i, testContext.rtkSet)
			g = testContext.evaluator.AddNew(g, g_rot)
		}

		beta_update := testContext.evaluator.AddNew(v, g)
		// U_j = S_j * M'' + Z2_j
		u := testContext.evaluator.MulRelinNew(s, m2, testContext.rlkSet)
		u = testContext.evaluator.AddNew(u, z2)

		// V+_j = eta * W_j + (1 - eta) * V_j + SumRowVec(U_j)
		v_update := testContext.evaluator.MulPtxtNew(v, const6_pt[a])
		beta = testContext.evaluator.MulPtxtNew(beta, const7_pt[a])
		v_update = testContext.evaluator.AddNew(v_update, beta)

		u = SumRowVec(testContext.evaluator, testContext.rtkSet, u, int(math.Pow(2, 9.0)), int(math.Pow(2, 5.0)))

		v_update = testContext.evaluator.AddNew(v_update, u)

		v = v_update
		beta = beta_update
	}
	end := time.Now()
	elapsed := end.Sub(start)
	fmt.Println("Training Time:", elapsed)

	// Inference in depth 3
	fmt.Println()
	fmt.Println("Inference...")
	correct := 0
	//for i := 0; i < numData; i++ {
	for i := 0; i < 3; i++ {
		if i%100 == 0 {
			fmt.Println(i, "-th Data Inferenced")
		}
		// Encrypt test data
		test_msg := mkckks.NewMessage(testContext.params)
		test_vector := make([]complex128, slotNum)
		for j := 0; j < numFeature; j++ {
			test_vector[j] = featureData[i][j]
		}
		test_msg.Value = test_vector
		test_ct := testContext.encryptor.EncryptMsgNewExpand(test_msg, testContext.pkSet.GetPublicKey(id), idset)

		// Compute inner product
		inner_prod := testContext.evaluator.MulRelinNew(test_ct, beta, testContext.rlkSet)

		for i := 1; i < numFeature; i *= 2 {
			inner_prod_rot := testContext.evaluator.RotateNew(inner_prod, i, testContext.rtkSet)
			inner_prod = testContext.evaluator.AddNew(inner_prod, inner_prod_rot)
		}

		// Compute sigmoid
		// term1 = inner_prod * inner_prod + c1/c3
		term1 := testContext.evaluator.MulRelinNew(inner_prod, inner_prod, testContext.rlkSet)
		term1 = testContext.evaluator.AddNew(term1, const5_ct)

		// term2 = c3 * inner_prod
		term2 := testContext.evaluator.MulPtxtNew(inner_prod, const8_pt)

		// sigmoid = term1 * term2 + c0
		sigmoid := testContext.evaluator.MulRelinNew(term1, term2, testContext.rlkSet)
		sigmoid = testContext.evaluator.AddNew(sigmoid, const9_ct)

		// Decrypt
		result := testContext.decryptor.Decrypt(sigmoid, testContext.skSet)
		if real(result.Value[0]) >= 0.5 && real(labelData[i]) == 1 {
			correct += 1
		} else if real(result.Value[0]) < 0.5 && real(labelData[i]) == 0 {
			correct += 1
		}
	}
	fmt.Println("Correct:", correct)

	/*
		// Check by Index
		correct := 0
		i := 1935
		// Encrypt test data
		fmt.Println()
		fmt.Println("Test Data ID:", i)
		fmt.Println("Encrypting Test Data...")
		test_msg := mkckks.NewMessage(testContext.params)
		test_vector := make([]complex128, slotNum)
		for j := 0; j < numFeature; j++ {
			test_vector[j] = featureData[i][j]
		}
		test_msg.Value = test_vector
		test_ct := testContext.encryptor.EncryptMsgNewExpand(test_msg, testContext.pkSet.GetPublicKey(id), idset)

		fmt.Println("Inference...")
		start = time.Now()
		// Compute inner product
		inner_prod := testContext.evaluator.MulRelinNew(test_ct, beta, testContext.rlkSet)

		for i := 1; i < numFeature; i *= 2 {
			inner_prod_rot := testContext.evaluator.RotateNew(inner_prod, i, testContext.rtkSet)
			inner_prod = testContext.evaluator.AddNew(inner_prod, inner_prod_rot)
		}

		// Compute sigmoid
		// term1 = inner_prod * inner_prod + c1/c3
		term1 := testContext.evaluator.MulRelinNew(inner_prod, inner_prod, testContext.rlkSet)
		term1 = testContext.evaluator.AddNew(term1, const5_ct)

		// term2 = c3 * inner_prod
		term2 := testContext.evaluator.MulPtxtNew(inner_prod, const8_pt)

		// sigmoid = term1 * term2 + c0
		sigmoid := testContext.evaluator.MulRelinNew(term1, term2, testContext.rlkSet)
		sigmoid = testContext.evaluator.AddNew(sigmoid, const9_ct)

		end = time.Now()
		elapsed = end.Sub(start)
		fmt.Println("Inference Time:", elapsed)

		fmt.Println("Decrypting Result...")
		// Decrypt
		result := testContext.decryptor.Decrypt(sigmoid, testContext.skSet)
		if real(result.Value[0]) >= 0.5 && real(labelData[i]) == 1 {
			correct += 1
		} else if real(result.Value[0]) < 0.5 && real(labelData[i]) == 0 {
			correct += 1
		}

		fmt.Println("Correct?", correct)
	*/
	return
}

//////////////////////////////////////////////////////
///////  Functions for logistic regression  /////////
/////////////////////////////////////////////////////

func SumRowVec(eval *mkckks.Evaluator, rtkSet *mkrlwe.RotationKeySet, ctIn *mkckks.Ciphertext, numRow, numCol int) (ctOut *mkckks.Ciphertext) {
	ctOut = ctIn
	for i := 1; i < numRow; i *= 2 {
		ctOut_rot := eval.RotateNew(ctOut, numCol*i, rtkSet)
		ctOut = eval.AddNew(ctOut, ctOut_rot)
	}
	return
}

func SumColVec(eval *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet, rtkSet *mkrlwe.RotationKeySet, ctIn *mkckks.Ciphertext, mask *ckks.Plaintext, numRow, numCol int) (ctOut *mkckks.Ciphertext) {
	ctOut = ctIn
	for i := 1; i < numCol; i *= 2 {
		ctOut_rot := eval.RotateNew(ctOut, i, rtkSet)
		ctOut = eval.AddNew(ctOut, ctOut_rot)
	}
	ctOut = eval.MulPtxtNew(ctOut, mask)

	for i := 1; i < numCol; i *= 2 {
		ctOut_rot := eval.RotateNew(ctOut, -i, rtkSet)
		ctOut = eval.AddNew(ctOut, ctOut_rot)
	}
	return
}

func genTestParams(defaultParam mkckks.Parameters, idset *mkrlwe.IDSet) (testContext *testParams, err error) {
	testContext = new(testParams)

	testContext.params = defaultParam

	rots := []int{14, 15, 384, 512, 640, 768, 896, 8191, 8190, 8188, 8184}

	for _, rot := range rots {
		testContext.params.AddCRS(rot)
	}

	testContext.kgen = mkckks.NewKeyGenerator(testContext.params)

	testContext.skSet = mkrlwe.NewSecretKeySet()
	testContext.pkSet = mkrlwe.NewPublicKeyKeySet()
	testContext.rlkSet = mkrlwe.NewRelinearizationKeyKeySet(defaultParam.Parameters)
	testContext.rtkSet = mkrlwe.NewRotationKeySet()

	for i := 0; i < testContext.params.LogN()-1; i++ {
		rots = append(rots, 1<<i)
	}

	for id := range idset.Value {
		sk, pk := testContext.kgen.GenKeyPair(id)
		r := testContext.kgen.GenSecretKey(id)
		rlk := testContext.kgen.GenRelinearizationKey(sk, r)

		for _, rot := range rots {
			rk := testContext.kgen.GenRotationKey(rot, sk)
			testContext.rtkSet.AddRotationKey(rk)
		}

		testContext.skSet.AddSecretKey(sk)
		testContext.pkSet.AddPublicKey(pk)
		testContext.rlkSet.AddRelinearizationKey(rlk)
	}

	testContext.ringQ = defaultParam.RingQ()

	if testContext.prng, err = utils.NewPRNG(); err != nil {
		return nil, err
	}

	testContext.encryptor = mkckks.NewEncryptor(testContext.params)
	testContext.decryptor = mkckks.NewDecryptor(testContext.params)
	testContext.evaluator = mkckks.NewEvaluator(testContext.params)

	return testContext, nil
}

func readData(filename string) ([][]complex128, []complex128) {
	feature_data := make([][]complex128, numData)
	label_data := make([]complex128, numData)
	f, err1 := os.Open(filename)
	if err1 != nil {
		panic(err1)
	}

	reader := csv.NewReader(bufio.NewReader(f))
	rows, err2 := reader.ReadAll()
	if err2 != nil {
		panic(err2)
	}

	for i, row := range rows {
		if i == 0 {
			continue
		}
		feature_data[i-1] = make([]complex128, numFeature)
		for j := 0; j < len(row); j++ {
			real_part, err3 := strconv.ParseFloat(row[j], 64)
			if err3 != nil {
				panic(err3)
			}
			if j != len(row)-1 {
				feature_data[i-1][j] = complex(real_part, 0)
			} else {
				label_data[i-1] = complex(real_part, 0)
			}
		}
	}
	return feature_data, label_data
}
