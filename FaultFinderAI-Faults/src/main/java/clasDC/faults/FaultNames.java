package clasDC.faults;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import lombok.Getter;

/**
 * This enum is used to deal with fault names in a readable manner.
 *
 * Each fault has its designated name and a number that represents its index in
 * the label.
 */
@Getter
public enum FaultNames {

	PIN_SMALL(
			0,
			"pin_small",
			new double[][] { { 8.0, 1.0 } },
			1.0,
			"pin",
			Nd4j.zeros(6, 10),
			new FaultSlidingInformation()),
	PIN_BIG(
			1,
			"pin_big",
			new double[][] { { 16.0, 1.0 } },
			2.0,
			"pin",
			Nd4j.zeros(6, 2),
			new FaultSlidingInformation()),
	CHANNEL_ONE(
			2,
			"Channel_1",
			new double[][] { { 8.0, 6.0 } },
			3.0,
			"channel",
			Nd4j.zeros(1, 4),
			new FaultSlidingInformation()),
	CHANNEL_TWO(
			3,
			"Channel_2",
			new double[][] { { 16.0, 6.0 } },
			4.0,
			"channel",
			Nd4j.zeros(1, 3),
			new FaultSlidingInformation()),
	CHANNEL_THREE(
			4,
			"Channel_3",
			new double[][] { { 32.0, 6.0 } },
			5.0,
			"channel",
			Nd4j.zeros(1, 1),
			new FaultSlidingInformation()),
	CONNECTOR_E(
			5,
			"Connector_E",
			new double[][] { { 3.0, 6.0 } },
			6.0,
			"connector",
			Nd4j.zeros(1, 14),
			new FaultSlidingInformation()),
	CONNECTOR_TREE(
			6,
			"Connector_Tree",
			new double[][] { { 3.0, 6.0 } },
			7.0,
			"connector",
			Nd4j.zeros(1, 14),
			new FaultSlidingInformation()),
	CONNECTOR_THREE(
			7,
			"Connector_Three",
			new double[][] { { 3.0, 6.0 } },
			8.0,
			"connector",
			Nd4j.zeros(1, 14),
			new FaultSlidingInformation()),
	FUSE_A(8, "Fuse_A", new double[][] { { 6.0, 6.0 } }, 9.0, "fuse", Nd4j.zeros(1, 7), new FaultSlidingInformation()),
	FUSE_B(9, "Fuse_B", new double[][] { { 6.0, 6.0 } }, 10.0, "fuse", Nd4j.zeros(1, 7), new FaultSlidingInformation()),
	FUSE_C(
			10,
			"Fuse_C",
			new double[][] { { 6.0, 6.0 } },
			11.0,
			"fuse",
			Nd4j.zeros(1, 7),
			new FaultSlidingInformation()),
	DEADWIRE(
			11,
			"Deadwire",
			new double[][] { { 1.0, 1.0 } },
			12.0,
			"wire",
			Nd4j.zeros(6, 112),
			new FaultSlidingInformation()),
	HOTWIRE(
			12,
			"Hotwire",
			new double[][] { { 1.0, 1.0 } },
			13.0,
			"wire",
			Nd4j.zeros(6, 112),
			new FaultSlidingInformation()),
	NOFAULT(
			13,
			"No_Fault",
			new double[][] { { 0.0, 0.0 } },
			14.0,
			"NA",
			Nd4j.zeros(1, 1),
			new FaultSlidingInformation());

	private final int index;
	private final String saveName;
	private final double[][] prior;
	private final double label;
	private final String familyOf;
	private final INDArray possiblePositions;
	// slidingInformation
	private final FaultSlidingInformation fSlidingInformation;

	FaultNames(int index, String saveName, double[][] prior, double label, String familyOf, INDArray possiblePositions,
			FaultSlidingInformation fSlidingInformation) {
		this.index = index;
		this.saveName = saveName;
		this.prior = prior;
		this.label = label;
		this.familyOf = familyOf;
		this.possiblePositions = possiblePositions;
		this.fSlidingInformation = fSlidingInformation;
		fSlidingInformation.setFaultName(this);

	}

	public String toString() {
		return this.saveName;
	}

	public static void main(String args[]) {
		System.out.println(Arrays.toString(FaultNames.PIN_SMALL.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.PIN_SMALL.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.PIN_BIG.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.PIN_BIG.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.CHANNEL_ONE.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.CHANNEL_ONE.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.CONNECTOR_E.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.CONNECTOR_E.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.CONNECTOR_TREE.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.CONNECTOR_TREE.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.CONNECTOR_THREE.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.CONNECTOR_THREE.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.FUSE_A.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.FUSE_A.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.FUSE_B.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.FUSE_B.getFSlidingInformation().getXEnd()));

		System.out.println(Arrays.toString(FaultNames.FUSE_C.getFSlidingInformation().getXStart()));
		System.out.println(Arrays.toString(FaultNames.FUSE_C.getFSlidingInformation().getXEnd()));

		int xMin = 0;
		int xMax = 12;

		int[] xStart = new int[10];
		int[] xEnd = new int[10];
		for (int i = 0; i < 10; i++) {
			xStart[i] = xMin;
			xEnd[i] = xMax;

			xMin += 8;
			xMax += 8;

		}

		// System.out.println(Arrays.toString(xStart));
		// System.out.println(Arrays.toString(xEnd));

	}
}
