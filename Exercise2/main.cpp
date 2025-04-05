#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

using namespace Eigen;
using namespace std;

// decomposizione PALU
Vector2d PALU(const Matrix2d& A, const Vector2d& b) {
    Vector2d x = A.fullPivLu().solve(b);
    return x;
}
// decomposizione QR
Vector2d QR(const Matrix2d& A, const Vector2d& b) {
    Vector2d x = A.householderQr().solve(b);
    return x;
}

// Funzione per calcolare l'errore relativo
double ErroreRelativo(const Vector2d& x_calcolato, const Vector2d& x_esatto) {
    return (x_calcolato - x_esatto).norm() / x_esatto.norm();
}

int main() {
    cout << fixed << setprecision(1) << scientific;
    
    Vector2d x(-1.0, -1.0);
    
    // sistema 1
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);
    
    Vector2d x1p = PALU(A1, b1);
    cout << "La soluzione (PALU) del sistema 1 è: [" << x1p.transpose() << "]" << endl;
    cout << "Errore relativo (PALU) del sistema 1: " << ErroreRelativo(x1p, x) << "\n" << endl;
        
    Vector2d x1q = QR(A1, b1);
    cout << "La soluzione (QR) del sistema 1 è: [" << x1q.transpose() << "]" << endl;
    cout << "Errore relativo (QR) del sistema 1: " << ErroreRelativo(x1q, x) << "\n" << endl;
        

    // sistema 2
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);
    
    Vector2d x2p = PALU(A2, b2);
    cout << "La soluzione (PALU) del sistema 2 è: [" << x2p.transpose() << "]" << endl;
    cout << "Errore relativo (PALU) del sistema 2: " << ErroreRelativo(x2p, x) <<  "\n" << endl;
    
    Vector2d x2q = QR(A2, b2);
    cout << "La soluzione (QR) del sistema 2 è: [" << x2q.transpose() << "]" << endl;
    cout << "Errore relativo (QR) del sistema 2: " << ErroreRelativo(x2q, x) <<  "\n" << endl;
    
    // sistema 3
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);
    
    Vector2d x3p = PALU(A3, b3);
    cout << "La soluzione (PALU) del sistema 3 è: [" << x3p.transpose() << "]" << endl;
    cout << "Errore relativo (PALU) del sistema 3: " << ErroreRelativo(x3p, x) <<  "\n" << endl;
    
    Vector2d x3q = QR(A3, b3);
    cout << "La soluzione (QR) del sistema 3 è: [" << x3q.transpose() << "]" << endl;
    cout << "Errore relativo (QR) del sistema 3: " << ErroreRelativo(x3q, x) <<  "\n" << endl;
    
    return 0;
}
