import java.io.*;
 
class test2{
public static void main(String a[]) throws IOException {
 
	String prg = "import sys\na=69\nprint(a)\n";
	BufferedWriter out = new BufferedWriter(new FileWriter("test1.py"));
	out.write(prg);
	out.close();
 
	ProcessBuilder pb = new ProcessBuilder("python","test1.py");
	Process p = pb.start();
 
	BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
	String ret = new String(in.readLine());
	System.out.println("value is : "+ ret);
}
}
