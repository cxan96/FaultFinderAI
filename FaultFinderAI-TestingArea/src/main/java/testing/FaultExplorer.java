package testing;

import java.util.Random;
import java.util.Scanner;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.NDArrayUtil;

import faulttypes.FaultData;
import faulttypes.FaultFactory;

public class FaultExplorer {
	public static void main(String args[]) {

		FaultFactory factory = new FaultFactory();

		Scanner sc = new Scanner(System.in);

		Random r = new Random();

		for (int i = 0; i < 50; i++) {
			int type = i % 7;
			FaultData fault = factory.getFault(type);
			INDArray labelArray = NDArrayUtil.toNDArray(factory.getReducedLabel());
			System.out.println(labelArray);
			System.out.println(type);
			fault.plotData();
			sc.nextLine();
		}
	}
}
