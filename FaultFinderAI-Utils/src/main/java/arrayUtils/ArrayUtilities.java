package arrayUtils;

public class ArrayUtilities {

	private ArrayUtilities() {

	}

	public static int nLayers = 6;
	public static int nWires = 112;
	public static int[] hvPinFault = new int[72];
	public static int[] hvChannelFault = new int[8];
	public static int[] hvConnectorFault = new int[42];
	public static int[] hvFuseFault = new int[21];
	public static int[] hvWireFault = new int[112 * 6];

	public static int faultLableSize = hvPinFault.length + hvChannelFault.length + hvConnectorFault.length
			+ hvFuseFault.length + 2 * hvWireFault.length;

}
