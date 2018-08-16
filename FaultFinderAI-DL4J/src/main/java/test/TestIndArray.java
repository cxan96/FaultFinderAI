package test;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestIndArray {

	public static void main(String[] args) {
		INDArray array = Nd4j.zeros(6);

		List<Integer> aList = new ArrayList<>();
		aList.add(2);
		aList.add(3);
		aList.add(4);
		aList.add(5);
		aList.add(5);
		aList.add(6);

		array.putScalar(0, 2);
		array.putScalar(1, 3);
		array.putScalar(2, 4);
		array.putScalar(3, 5);
		array.putScalar(4, 5);
		array.putScalar(5, 6);

		// for (int i = 0; i < array.length(); i++) {
		// System.out.println(array.getDouble(i));
		// }

		INDArray array2 = Nd4j.create(aList);
		System.out.println(array.medianNumber() + "   " + array2.medianNumber() + "  " + array2.length() + "  "
				+ array2.meanNumber());
	}
}
