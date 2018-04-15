import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

class Rohs {
    public static void main(String[] args) throws IOException, InterruptedException {
        // https://stackoverflow.com/questions/1401002/trick-an-application-into-thinking-its-stdin-is-interactive-not-a-pipe
        //
        // Using script or unbuffer is the important catch. Without this
        // step you cannot use stdin in Python interactively, even with
        // python -u. At least script comes with Linux/Mac OS X, but
        // unbuffer works fine too.
        ProcessBuilder pb;
        switch(System.getProperty("os.name")) {
            case "Mac OS X":
                pb = new ProcessBuilder(
                    "/usr/bin/script", "-q", "/dev/null", "/usr/bin/python");
                break;
            default:
                // Linux
                pb = new ProcessBuilder(
                    "/usr/bin/script", "-qfc", "/usr/bin/python", "/dev/null");

        }
        // This doesn't make a difference.
        // pb.redirectErrorStream(true);

        Process p = pb.start();

        char[] readBuffer = new char[1000];
        InputStreamReader isr = new InputStreamReader(p.getInputStream());
        BufferedReader br = new BufferedReader(isr);
        int charCount;
        boolean written = false;
        while(true) {
            if (!br.ready() && !written) {
                // Ugly. Should be reading for '>>>' prompt then writing.
                Thread.sleep(1000);
                if (!written) {
                    written = true;
                    OutputStream os = p.getOutputStream();
                    OutputStreamWriter osw = new OutputStreamWriter(os);
                    BufferedWriter bw = new BufferedWriter(osw);
                    bw.write("2+2");
                    bw.newLine();
                    bw.write("quit()");
                    bw.newLine();
                    bw.flush();
                    bw.close();
                }
                continue;
            }
            charCount = br.read(readBuffer);
            if (charCount > 0)
                System.out.print(new String(readBuffer, 0, charCount));
            else
                break;
        }
    }
}
