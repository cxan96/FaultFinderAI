package utils;

public class ArrayUtilities {

	private ArrayUtilities() {

	}

	public static int nLayers = 6;
	public static int nWires = 112;
	public static int[] hvPinFault = new int[72];
	public static int[] hvChannelFault = new int[8];
	public static int[] hvConnectorFault = new int[42];
	public static int[] hvFuseFault = new int[21];
	public static int[] hvHotWireFault = new int[112 * 6];
	public static int[] hvDeadWireFault = new int[112 * 6];
	public static int[] hvNoWireFault = new int[112 * 6];

	public static int faultLableSize = hvPinFault.length + hvChannelFault.length + hvConnectorFault.length
			+ hvFuseFault.length + 2 * hvNoWireFault.length;

	// reduced labels
	// might be interesting for CNN's

	public static int reducedfaultLableSize = hvPinFault.length + hvChannelFault.length + hvConnectorFault.length
			+ hvFuseFault.length + 3 * hvNoWireFault.length;
	public static int[] hvReducedPinFault = new int[2];
	public static int[] hvReducedChannelFault = new int[3];
	public static int[] hvReducedConnectorFault = new int[3];
	public static int[] hvReducedFuseFault = new int[3];
	public static int[] hvReducedHotWireFault = new int[1];
	public static int[] hvReducedDeadWireFault = new int[1];
	public static int[] hvReducedNoFault = new int[1];

}
