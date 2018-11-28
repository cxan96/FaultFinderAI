package clasDC.faults;

import java.util.List;

import lombok.Builder;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
@Builder()
public class FaultFactory2 {
	@Builder.Default
	private final int nChannels = 3;

	private final List<Fault> faultList;

	/**
	 * superLayer is used to call the correct background data, This data has been
	 * engineered from actual data in Run 3923
	 */
	private final int superLayer;

	/**
	 * data is an array to generate faults into. Its ideal for each superLayer
	 */
	private final int[][] data;
	/**
	 * retData is an array to of faults that is returned in CLAS the coordinate <br>
	 * system of x = wires y = layers
	 */
	private final int[][] retData;
	/**
	 * List<Double> lMinMax is a list that contains the data minimum value and
	 * maximum value
	 */
	private final List<Integer> lMinMax;
	/**
	 * randomSuperlayer is whether or not to randomize the superlayer or use the
	 * user selected superlayer idea
	 */
	@Builder.Default
	private final boolean randomSuperlayer = true;

	/**
	 * nFaults is used to generate the number of background faults to differentiate
	 * against
	 */
	private final int nFaults;

	/**
	 * randomSmear is to blurr out the faults by the median value of the activations
	 * from the surrounding neighbors
	 */

	@Builder.Default
	private final boolean randomSmear = true;

	/**
	 * desiredFault is used to check if the fault to learn from was generated
	 */

	@Builder.Default
	private final FaultNames desiredFault = FaultNames.CHANNEL_ONE;

	@Builder.Default
	private final int maxFaults = 10;

	/**
	 * 
	 * 
	 * I cannot get things to initialize, will check back later
	 * 
	 * @param args
	 */

//	private FaultFactory2() {
//		this.nFaults = ThreadLocalRandom.current().nextInt(0, maxFaults + 1);
//	}

	public static void main(String[] args) {
		FaultFactory2 f = FaultFactory2.builder().nChannels(6).build();
		System.out.println(f.getNChannels() + "  " + f.isRandomSuperlayer() + "  " + f.getNFaults());
	}

}// end
	// of
	// FaultFactory
	// class.
