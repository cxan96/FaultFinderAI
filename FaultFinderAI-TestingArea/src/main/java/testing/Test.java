package testing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import utils.FaultUtils;

public class Test {
	private static double yDiv = 5. / 6.;
	private static double xDiv = 111. / 112.;
	public static double[][] HVChannelPriors = { { 8.0, 6.0 }, { 16.0, 6.0 }, { 32.0, 6.0 } };

	public static void main(String[] args) {

		// double[][] HVChannelPriors = { { 8.0, 6.0 }, { 16.0, 6.0 }, { 32.0,
		// 6.0 } };

		double[][] divide = { { 2, 3 } };

		double[][] priors = FaultUtils.allPriors;
		double[][] check = FaultUtils.getPriors(new double[][] { { 2, 3 } });

		for (int i = 0; i < priors.length; i++) {
			for (int j = 0; j < priors[0].length; j++) {
				System.out.println(priors[i][j] + " " + i + " " + j + " " + divide[0][j] + " " + check[i][j]);

			}
		}
		INDArray priorINDs = Nd4j.create(HVChannelPriors);
		INDArray quot = Nd4j.create(divide);

	}

}
