package utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import javax.swing.JOptionPane;

public class DomainUtils {

	private static String getHostName() {
		String retString = null;
		try {
			Process p = Runtime.getRuntime().exec("hostname");

			BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
			retString = stdInput.readLine();

		} catch (IOException e) {
			System.out.println("exception happened - here's what I know: ");
			e.printStackTrace();
			System.exit(-1);
		}
		return retString;
	}

	public static String getDataLocation() {
		String hostname = getHostName();
		if (hostname.contains("ikp")) {
			return "/Users/michaelkunkel/WORK/CLAS/CLAS12/CLAS12Data/RGACooked/V5b.2.1/";
		} else if (hostname.contains("MichaelKunkel")) {
			return "/Users/Mike/Google\\ Drive/CLAS12Data/";
		} else {
			JOptionPane.showMessageDialog(null, "Need to set this in DomainUtils", "Your Fault",
					JOptionPane.ERROR_MESSAGE);
			return "";
		}
	}
}
