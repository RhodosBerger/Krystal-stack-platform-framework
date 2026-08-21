import React, { createContext, useContext, useState, useEffect } from 'react';

// Default to Operator for safety
const initialAuth = {
  user: {
    username: 'guest',
    role: 'OPERATOR', // 'OPERATOR', 'MANAGER', 'ENGINEER' (Creator), 'ADMIN'
    token: null
  },
  isAuthenticated: false
};

const AuthContext = createContext(initialAuth);

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [auth, setAuth] = useState(initialAuth);

  useEffect(() => {
    // 1. Check URL Params (Magic Link)
    const params = new URLSearchParams(window.location.search);
    const urlRole = params.get('role');

    // 2. Check LocalStorage
    const savedRole = localStorage.getItem('dev_role');

    const validRoles = ['OPERATOR', 'MANAGER', 'ENGINEER', 'ADMIN'];
    let targetRole = null;

    if (urlRole && validRoles.includes(urlRole.toUpperCase())) {
        targetRole = urlRole.toUpperCase();
        // Clean URL without refresh
        window.history.replaceState({}, document.title, window.location.pathname);
    } else if (savedRole && validRoles.includes(savedRole)) {
        targetRole = savedRole;
    }

    if (targetRole) {
      setAuth({
        user: { username: 'dev_user', role: targetRole },
        isAuthenticated: true
      });
      localStorage.setItem('dev_role', targetRole);
    }
  }, []);

  const login = (role) => {
    const user = { username: 'dev_user', role };
    setAuth({ user, isAuthenticated: true });
    localStorage.setItem('dev_role', role);
  };

  const logout = () => {
    setAuth(initialAuth);
    localStorage.removeItem('dev_role');
  };

  const hasPermission = (requiredRole) => {
    const roles = ['OPERATOR', 'MANAGER', 'ENGINEER', 'ADMIN'];
    const userRoleIdx = roles.indexOf(auth.user.role);
    const requiredRoleIdx = roles.indexOf(requiredRole);
    return userRoleIdx >= requiredRoleIdx;
  };

  return (
    <AuthContext.Provider value={{ ...auth, login, logout, hasPermission }}>
      {children}
    </AuthContext.Provider>
  );
};
