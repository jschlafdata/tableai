const sidebarStyles = {
  default: {
    sidebar: {
      container: {},
      width: 280,
      collapsedWidth: 95
    },
    menuItem: {
      root: { px: 2, py: 1 },
      button: ({ active }) => ({
        borderRadius: 8,
        padding: '6px 12px',
        margin: '8px 8px',
        backgroundColor: active ? 'rgba(110, 77, 219, 0.23)' : 'transparent',
        '&:hover': {
          backgroundColor: 'rgba(0, 0, 0, 0.04)'
        },
        transition: 'background-color 0.2s ease'
      }),
      // Add specific text styles
      text: {
        fontFamily: '"Poppins", "Roboto", sans-serif', // Change to your preferred font
        fontSize: '16px',
        fontWeight: 500,
        paddingLeft: '12px', // Add padding between icon and text
        color: '#555555'
      },
      // Active text styling
      activeText: {
        color: '#312E31',
        fontWeight: 600
      },
      // Submenu text style
      subMenuText: {
        fontSize: '13px',
        fontWeight: 500,
        color: '#666666'
      }
    },
    localIcon: ({ active }) => ({
      width: '50px',
      height: '50px', 
      objectFit: 'contain',
      opacity: active ? 1 : 0.8,
      filter: active ? 'none' : 'grayscale(30%)',
      transition: 'all 0.2s ease',
    }),
    // Section title styles
    sectionTitle: {
      fontFamily: '"Poppins", "Roboto", sans-serif',
      fontSize: '12px',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      color: '#888888',
      padding: '16px 24px 8px'
    },
    badge: (item) => ({
      ml: 1,
      '& .MuiBadge-badge': {
        backgroundColor: item.badgeColor || '#88e2e2',
        color: item.badgeTextColor || '#333',
        fontSize: 10,
        height: 20,
        minWidth: 20
      }
    }),
    icon: ({ active }) => ({
      fontSize: 24, // Slightly smaller than your current 30
      color: active ? '#312E31' : '#767676'
    }),
    logo: (collapsed) => ({
      height: 32,
      width: 'auto',
      mr: collapsed ? 0 : 1
    }),
    toggleButton: {
      width: 32,
      height: 32,
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
      backgroundColor: '#fff',
      ml: 1,
      mr: 3
    },
    content: {
      main: {
        flexGrow: 1,
        p: 3,
        bgcolor: '#ffffff',
        overflowY: 'auto'
      },
      header: {
        height: 64,
        px: 2,
        bgcolor: 'transparent',
        color: '#333',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        minWidth: 95
      }
    }
  }
};

export default sidebarStyles;