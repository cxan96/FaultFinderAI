package test;

import java.util.Arrays;

public class ATest {
	public static void main(String[] args) {
		double[] d = { 0.0, 0.0, 3.5779922745244145E-30, 3.528650300671787E-39, 0.9632607102394104, 0.0, 0.0, 0.0, 0.0,
				2.6638375232373034E-33, 0.0, 0.0, 0.03673921152949333, 3.065711240424207E-8 };
		double[] blah = Arrays.stream(d).map(x -> (x < 1E-04) ? 0.0 : x).toArray();
		// List<Integer> collect1 = num.stream().map(n -> n *
		// 2).collect(Collectors.toList());
		// d = Arrays.stream(d).filter(x -> x > 1E-4).toArray();
		System.out.println(Arrays.toString(d));

		System.out.println(Arrays.toString(blah));

	}
}
