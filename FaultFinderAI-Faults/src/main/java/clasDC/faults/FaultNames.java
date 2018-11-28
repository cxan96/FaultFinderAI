package clasDC.faults;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import lombok.Getter;
import utils.FaultUtils;

/**
 * This enum is used to deal with fault names in a readable manner.
 *
 * Each fault has its designated name and a number that represents its index in
 * the label.
 */
@Getter
public enum FaultNames {
	PIN_SMALL("pin_small", 0, "pin_small", new double[][] { { 8.0, 1.0 } }), PIN_BIG("pin_big", 1, "pin_big",
			new double[][] { { 16.0, 1.0 } }), CHANNEL_ONE("Channel 1", 2, "Channel_1", new double[][] { { 8.0,
					6.0 } }), CHANNEL_TWO("Channel 2", 3, "Channel_2", new double[][] { { 16.0, 6.0 } }), CHANNEL_THREE(
							"Channel 3", 4, "Channel_3", new double[][] { { 32.0, 6.0 } }), CONNECTOR_E("Connector E",
									5, "Connector_E", new double[][] { { 3.0, 6.0 } }), CONNECTOR_TREE("Connector Tree",
											6, "Connector_Tree", new double[][] { { 3.0, 6.0 } }), CONNECTOR_THREE(
													"Connector Three", 7, "Connector_Three",
													new double[][] { { 3.0, 6.0 } }), FUSE_A("Fuse A", 8, "Fuse_A",
															new double[][] { { 6.0, 6.0 } }), FUSE_B("Fuse B", 9,
																	"Fuse_B", new double[][] { { 6.0, 6.0 } }), FUSE_C(
																			"Fuse C", 10, "Fuse_C",
																			new double[][] { { 6.0, 6.0 } }), DEADWIRE(
																					"deadwire", 11, "Deadwire",
																					new double[][] {
																							{ 1.0, 1.0 } }), HOTWIRE(
																									"hotwire", 12,
																									"Hotwire",
																									new double[][] {
																											{ 1.0, 1.0 } }), NOFAULT(
																													"No Fault",
																													13,
																													"No_Fault",
																													new double[][] {
																															{ 0.0, 0.0 } });

	private final String name;
	private final int index;
	private final String saveName;
	private final double[][] prior;

	FaultNames(String name, int index, String saveName) {
		this(name, index, saveName, new double[][] { {} });
	}

	FaultNames(String name, int index, String saveName, double[][] prior) {
		this.name = name;
		this.index = index;
		this.saveName = saveName;
		this.prior = prior;
	}

	public String toString() {
		return this.name;
	}

	public static void main(String args[]) {
		System.out.println(FaultNames.PIN_SMALL.getIndex() + ": " + FaultNames.PIN_SMALL + "  "
				+ FaultNames.PIN_SMALL.getSaveName());
		System.out.println(FaultNames.PIN_BIG.getIndex() + ": " + FaultNames.PIN_BIG);
		System.out.println(FaultNames.CHANNEL_ONE.getIndex() + ": " + FaultNames.CHANNEL_ONE + "  "
				+ FaultNames.CHANNEL_ONE.getSaveName());

		FaultNames aNames = FaultNames.CHANNEL_ONE;
		System.out.println(aNames.getSaveName());
		System.out.println("###########################################");
		Set<String> labelSet = new HashSet<>();

		for (FaultNames d : FaultNames.values()) {
			System.out.println(d.getSaveName() + "  " + Arrays.deepToString(d.getPrior()));
			labelSet.add(d.getSaveName());

		}
		double[][] test = FaultNames.CHANNEL_ONE.getPrior();

		double[][] newtest = FaultUtils.merge(test, FaultNames.CHANNEL_ONE.getPrior(),
				FaultNames.CHANNEL_TWO.getPrior());
		System.out.println(Arrays.deepToString(newtest));

	}
}
